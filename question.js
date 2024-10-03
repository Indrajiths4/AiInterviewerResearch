const readline = require('readline');
const { GoogleGenerativeAI } = require("@google/generative-ai");
require('dotenv').config();

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const genAI = new GoogleGenerativeAI(process.env.API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

async function generateContent(prompt) {
    try {
        const result = await model.generateContent(prompt);
        return result.response.text().trim();
    } catch (error) {
        console.error('Error generating content:', error);
        return "I couldn't generate content at this time. Let's move on.";
    }
}

async function askQuestion(context = "") {
    const prompt = `As an interviewer for an entry-level Python developer position, ask a single, concise question about Python basics or fundamentals. The question should be simple and appropriate for beginners. ${context}`;
    return generateContent(prompt);
}

async function generateFollowUp(question, answer) {
    const prompt = `
Previous question: ${question}
Candidate's answer: ${answer}

As the interviewer, ask a natural, conversational follow-up question that either:
1. Seeks clarification on the previous answer
2. Explores a related basic Python concept
3. Builds upon the candidate's demonstrated knowledge

Keep the question concise and suitable for an entry-level position.`;

    return generateContent(prompt);
}

async function generateReport(interviewHistory) {
    const prompt = `
You are an AI assistant tasked with analyzing a Python developer interview. Based on the following interview transcript, provide a concise report that includes:

1. Overall assessment of the candidate's Python knowledge
2. Strengths demonstrated during the interview
3. Areas for improvement
4. Recommendation for hiring (Strongly Recommend, Recommend, Consider, or Do Not Recommend)

Interview Transcript:
${interviewHistory.map(item => `Interviewer: ${item.question}\nCandidate: ${item.answer}`).join('\n\n')}

Provide your analysis in a clear, professional manner.`;

    return generateContent(prompt);
}

async function conductInterview() {
    let interviewHistory = [];
    let question = await askQuestion();

    for (let i = 0; i < 5; i++) {
        console.log(`\nInterviewer: ${question}`);

        const answer = await new Promise(resolve => {
            rl.question("You: ", resolve);
        });

        interviewHistory.push({ question, answer });

        if (i < 4) {
            question = await generateFollowUp(question, answer);
        }
    }

    console.log("\nThank you for participating in this interview simulation.");
    console.log("Generating interview report...");

    const report = await generateReport(interviewHistory);
    console.log("\nInterview Analysis Report:");
    console.log(report);

    rl.close();
}

console.log("Welcome to the Python Developer Interview Simulator!");
console.log("I'll ask you a series of questions about Python development. Please provide your answers as you would in a real interview.\n");

conductInterview();