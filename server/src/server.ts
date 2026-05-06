import express from "express";
import cors from "cors";
import { ChatVertexAI, VertexAIEmbeddings } from "@langchain/google-vertexai";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import type { Document } from "@langchain/core/documents";

import { config } from "./config.js";
import { connectToDatabase, collections } from "./database.js";

const app = express();
app.use(cors());

const router = express.Router();
router.use(express.json());

router.get("/", async (_, res) => {
    res.send("Welcome to the SmartStudy Tutor API 🎓");
});

// =============================================================================
// LLM and embeddings
// =============================================================================
const model = new ChatVertexAI({
    model: "gemini-2.5-flash",
    maxOutputTokens: 2048,
    temperature: 0.4,
    topP: 0.9,
    topK: 20,
});

// =============================================================================
// Vector store — same Atlas collection as the ingestion Cloud Function
// =============================================================================
await connectToDatabase();

const vectorStore = new MongoDBAtlasVectorSearch(
    new VertexAIEmbeddings({ model: "text-embedding-005" }),
    {
        collection: collections.context as any,
        indexName: "vector_index",
        textKey: "text",
        embeddingKey: "embedding",
    }
);

// k=4 for normal chat (focused), k=8 for quiz (broader coverage)
const chatRetriever = vectorStore.asRetriever({ k: 4 });
const quizRetriever = vectorStore.asRetriever({ k: 8 });

// =============================================================================
// Helpers
// =============================================================================

/**
 * Formats retrieved chunks with their source/page metadata so that the LLM
 * can cite them precisely in its responses.
 */
function formatDocsWithCitations(docs: Document[]): string {
    if (!docs || docs.length === 0) {
        return "(no relevant excerpts were retrieved from your study materials)";
    }
    return docs.map((doc, i) => {
        const source = doc.metadata?.source ?? "unknown";
        const pageLabel = doc.metadata?.page_label;
        const page = doc.metadata?.page;
        // PyPDFLoader stores `page` as a 0-indexed integer; `page_label` is the
        // actual label printed on the page (often the same as page+1 but not always).
        const pageDisplay = pageLabel
            ?? (page !== undefined && page !== null ? String(Number(page) + 1) : null);
        const pageStr = pageDisplay ? `, page ${pageDisplay}` : "";
        return `[Excerpt ${i + 1} — ${source}${pageStr}]\n${doc.pageContent}`;
    }).join("\n\n---\n\n");
}

type ChatTurn = ["human", string] | ["assistant", string];

function toLangchainMessages(history: ChatTurn[]) {
    return history.map(([role, content]) =>
        role === "human" ? new HumanMessage(content) : new AIMessage(content)
    );
}

// =============================================================================
// Tutor persona — Formal Academic Tutor
// =============================================================================
const TUTOR_SYSTEM_INSTRUCTION = `You are SmartStudy, a Formal Academic Tutor helping a university student prepare for their exams using their own lecture notes.

Your behaviour rules:
1. ANSWER ACCURATELY using ONLY the context provided below. If the answer is not in the context, say so explicitly. Do not use external knowledge and do not invent facts.
2. ALWAYS CITE your sources after every factual claim using this exact format: [filename.pdf, page X]. Multiple citations are encouraged.
3. SUMMARIZE complex ideas in clear, formal academic English.
4. END EVERY RESPONSE with one short pedagogical follow-up question that helps the student deepen their understanding — for example a clarifying question, a small reasoning challenge, or a request to explain a concept back.
5. If the student is off-topic, politely redirect them to their study material.

Default language: English. Switch to the student's language only if they explicitly write in another language.`;

// =============================================================================
// Chat chain (LCEL): retrieve → prompt → model → parse
// =============================================================================
const chatPrompt = ChatPromptTemplate.fromMessages([
    ["system", TUTOR_SYSTEM_INSTRUCTION],
    new MessagesPlaceholder("history"),
    ["human", `Question: {question}

Context from your lecture notes:
{context}`],
]);

const chatChain = RunnableSequence.from([
    {
        question: (input: { question: string; history: ChatTurn[] }) => input.question,
        history: (input: { question: string; history: ChatTurn[] }) =>
            toLangchainMessages(input.history),
        context: async (input: { question: string; history: ChatTurn[] }) => {
            const docs = await chatRetriever.invoke(input.question);
            return formatDocsWithCitations(docs);
        },
    },
    chatPrompt,
    model,
    new StringOutputParser(),
]);

// =============================================================================
// Quiz chain (LCEL) — triggered by /quiz [optional topic]
// =============================================================================
const QUIZ_SYSTEM_INSTRUCTION = `You are SmartStudy, generating a 5-question multiple-choice quiz for a student based on their lecture notes.

Generate exactly 5 distinct questions covering the most important concepts in the provided context. Each question MUST:
- have a clear, focused stem
- have exactly 4 answer options labelled A, B, C, D
- have exactly one correct answer
- include a brief (1-2 sentence) explanation
- include a citation in the format [filename.pdf, page X]

Use ONLY information from the context. If the context is too thin to produce 5 high-quality questions, generate fewer and explain why.

Format your response in Markdown, exactly like this:

**Quiz — 5 questions on your material**

**Question 1.** <stem>

- A) <option>
- B) <option>
- C) <option>
- D) <option>

**Correct answer:** <letter>
**Explanation:** <one or two sentences>
**Source:** [filename.pdf, page X]

---

(repeat for questions 2 to 5, separating each with a horizontal rule)

End with a short encouragement and an offer to discuss any answer in detail.`;

const quizPrompt = ChatPromptTemplate.fromMessages([
    ["system", QUIZ_SYSTEM_INSTRUCTION],
    ["human", `Topic focus: {topic}

Context from the student's lecture notes:
{context}`],
]);

const quizChain = RunnableSequence.from([
    {
        topic: (input: { topic: string }) =>
            input.topic || "the most important concepts in the material",
        context: async (input: { topic: string }) => {
            const query = input.topic || "main concepts and key ideas overview";
            const docs = await quizRetriever.invoke(query);
            return formatDocsWithCitations(docs);
        },
    },
    quizPrompt,
    model,
    new StringOutputParser(),
]);


const history: ChatTurn[] = [];
const MAX_HISTORY_TURNS = 20;

router.post("/messages", async (req, res) => {
    const message: string | undefined = req.body?.text;
    if (!message || typeof message !== "string") {
        return res.status(400).send({ error: "Message is required" });
    }

    // /quiz command detection (case-insensitive, with optional topic)
    const quizMatch = message.match(/^\s*\/quiz\b\s*(.*)$/is);
    const isQuiz = !!quizMatch;
    const quizTopic = quizMatch?.[1]?.trim() ?? "";

    // RAG toggle — defaults to true for SmartStudy (it's a study tutor)
    const rag: boolean = req.body?.rag !== false;

    try {
        let answer: string;

        if (isQuiz) {
            answer = await quizChain.invoke({ topic: quizTopic });
        } else if (rag) {
            answer = await chatChain.invoke({ question: message, history });
        } else {
            // Fallback: tutor persona without retrieval (RAG toggle off)
            const messages = [
                new SystemMessage(TUTOR_SYSTEM_INSTRUCTION
                    + "\n\n(Retrieval is currently disabled. Tell the student you are answering without their notes.)"),
                ...toLangchainMessages(history),
                new HumanMessage(message),
            ];
            const response = await model.invoke(messages);
            answer = String(response.content ?? "");
        }

        // Update conversation history (skip /quiz output to keep chat focused)
        if (!isQuiz) {
            history.push(["human", message]);
            history.push(["assistant", answer]);
            if (history.length > MAX_HISTORY_TURNS * 2) {
                history.splice(0, history.length - MAX_HISTORY_TURNS * 2);
            }
        }

        return res.send({ text: answer });
    } catch (e: any) {
        console.error("[/messages] error:", e?.message ?? e);
        return res.status(500).send({
            error: "I'm having trouble right now. Please try again in a moment.",
        });
    }
});

app.use(router);

app.listen(config.server.port, () => {
    console.log(`SmartStudy Tutor server running on port:${config.server.port}...`);
});
