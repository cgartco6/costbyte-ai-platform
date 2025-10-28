const { OpenAI } = require('openai');
const axios = require('axios');

class SyntheticIntelligence {
    constructor() {
        this.openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY
        });
        this.agents = new Map();
        this.taskQueue = [];
        this.isProcessing = false;
    }

    // Create AI Agent for specific task
    async createAgent(agentConfig) {
        const agent = {
            id: this.generateAgentId(),
            name: agentConfig.name,
            type: agentConfig.type,
            capabilities: agentConfig.capabilities,
            knowledge: agentConfig.knowledge || [],
            memory: [],
            createdAt: new Date(),
            status: 'active'
        };

        // Initialize agent with base knowledge
        if (agentConfig.initialPrompt) {
            await this.trainAgent(agent, agentConfig.initialPrompt);
        }

        this.agents.set(agent.id, agent);
        return agent;
    }

    // Train agent with specific knowledge
    async trainAgent(agent, trainingData) {
        const prompt = `
        You are an AI agent specialized in ${agent.type}. 
        Your capabilities include: ${agent.capabilities.join(', ')}
        
        Training Data: ${trainingData}
        
        Learn from this information and be ready to perform tasks related to your specialization.
        `;

        try {
            const response = await this.openai.chat.completions.create({
                model: "gpt-4",
                messages: [
                    {
                        role: "system",
                        content: prompt
                    },
                    {
                        role: "user",
                        content: "Acknowledge your training and confirm you're ready to perform tasks."
                    }
                ],
                temperature: 0.3,
                max_tokens: 500
            });

            agent.memory.push({
                type: 'training',
                data: trainingData,
                timestamp: new Date(),
                response: response.choices[0].message.content
            });

            return response.choices[0].message.content;
        } catch (error) {
            console.error('Error training agent:', error);
            throw new Error('Failed to train AI agent');
        }
    }

    // Execute complex task with AI agent
    async executeTask(agentId, task, context = {}) {
        const agent = this.agents.get(agentId);
        if (!agent) {
            throw new Error('Agent not found');
        }

        const taskId = this.generateTaskId();
        this.taskQueue.push({
            id: taskId,
            agentId,
            task,
            context,
            status: 'pending',
            createdAt: new Date()
        });

        // Process task
        return await this.processTask(taskId);
    }

    async processTask(taskId) {
        const task = this.taskQueue.find(t => t.id === taskId);
        if (!task) throw new Error('Task not found');

        const agent = this.agents.get(task.agentId);
        
        try {
            task.status = 'processing';
            
            // Analyze task complexity
            const complexity = await this.analyzeTaskComplexity(task.task);
            
            if (complexity === 'high') {
                return await this.processComplexTask(agent, task);
            } else {
                return await this.processSimpleTask(agent, task);
            }
        } catch (error) {
            task.status = 'failed';
            task.error = error.message;
            throw error;
        }
    }

    async processComplexTask(agent, task) {
        // Break down complex task into subtasks
        const subtasks = await this.breakdownTask(task.task);
        const results = [];

        for (const subtask of subtasks) {
            const result = await this.executeSubtask(agent, subtask, task.context);
            results.push(result);
            
            // Learn from execution
            await this.learnFromExecution(agent, subtask, result);
        }

        // Synthesize results
        const finalResult = await this.synthesizeResults(agent, task.task, results);
        
        task.status = 'completed';
        task.result = finalResult;
        task.completedAt = new Date();

        return finalResult;
    }

    async breakdownTask(taskDescription) {
        const prompt = `
        Break down the following complex task into smaller, manageable subtasks:
        
        TASK: ${taskDescription}
        
        Return the subtasks as a JSON array of objects, each with:
        - id: unique identifier
        - description: clear description of the subtask
        - dependencies: any dependencies on other subtasks
        - estimated_duration: time estimate in minutes
        
        Format the response as valid JSON only.
        `;

        const response = await this.openai.chat.completions.create({
            model: "gpt-4",
            messages: [
                {
                    role: "system",
                    content: "You are an expert at breaking down complex tasks into manageable subtasks."
                },
                {
                    role: "user",
                    content: prompt
                }
            ],
            temperature: 0.4,
            max_tokens: 2000
        });

        return JSON.parse(response.choices[0].message.content);
    }

    async executeSubtask(agent, subtask, context) {
        const prompt = `
        You are ${agent.name}, an AI agent specialized in ${agent.type}.
        Your capabilities: ${agent.capabilities.join(', ')}
        
        Current subtask: ${subtask.description}
        Context: ${JSON.stringify(context)}
        
        Previous knowledge from your memory may be relevant.
        
        Execute this subtask and provide the result.
        `;

        const response = await this.openai.chat.completions.create({
            model: "gpt-4",
            messages: [
                {
                    role: "system",
                    content: `You are ${agent.name}. ${agent.memory.map(m => m.data).join(' ')}`
                },
                {
                    role: "user",
                    content: prompt
                }
            ],
            temperature: 0.7,
            max_tokens: 1500
        });

        return {
            subtaskId: subtask.id,
            result: response.choices[0].message.content,
            timestamp: new Date()
        };
    }

    async learnFromExecution(agent, subtask, result) {
        agent.memory.push({
            type: 'execution',
            subtask: subtask.description,
            result: result.result,
            timestamp: new Date(),
            success: true
        });

        // Keep memory manageable
        if (agent.memory.length > 100) {
            agent.memory = agent.memory.slice(-50);
        }
    }

    async synthesizeResults(agent, originalTask, results) {
        const prompt = `
        Synthesize the following results from subtasks into a complete solution for the original task.
        
        ORIGINAL TASK: ${originalTask}
        
        SUBTASK RESULTS:
        ${results.map(r => `- ${r.subtaskId}: ${r.result}`).join('\n')}
        
        Provide a comprehensive, well-structured final result that addresses the original task completely.
        `;

        const response = await this.openai.chat.completions.create({
            model: "gpt-4",
            messages: [
                {
                    role: "system",
                    content: "You are an expert at synthesizing multiple task results into a cohesive solution."
                },
                {
                    role: "user",
                    content: prompt
                }
            ],
            temperature: 0.5,
            max_tokens: 2000
        });

        return response.choices[0].message.content;
    }

    // Analyze task complexity
    async analyzeTaskComplexity(taskDescription) {
        const prompt = `
        Analyze the complexity of this task and classify it as 'low', 'medium', or 'high':
        
        TASK: ${taskDescription}
        
        Consider:
        - Number of steps required
        - Domain knowledge needed
        - Potential challenges
        - Time estimation
        
        Respond with only one word: low, medium, or high.
        `;

        const response = await this.openai.chat.completions.create({
            model: "gpt-4",
            messages: [
                {
                    role: "system",
                    content: "You are a task complexity analyzer."
                },
                {
                    role: "user",
                    content: prompt
                }
            ],
            temperature: 0.3,
            max_tokens: 10
        });

        return response.choices[0].message.content.trim().toLowerCase();
    }

    // Multi-agent collaboration
    async collaborativeTaskExecution(taskDescription, requiredAgents) {
        const subtasks = await this.breakdownTask(taskDescription);
        const agentAssignments = await this.assignSubtasksToAgents(subtasks, requiredAgents);
        
        const results = [];
        
        for (const assignment of agentAssignments) {
            const agent = this.agents.get(assignment.agentId);
            const result = await this.executeSubtask(agent, assignment.subtask, {});
            results.push(result);
        }

        return await this.synthesizeCollaborativeResults(taskDescription, results, agentAssignments);
    }

    async assignSubtasksToAgents(subtasks, availableAgents) {
        const assignments = [];
        
        for (const subtask of subtasks) {
            // Find best agent for subtask
            const bestAgent = await this.findBestAgentForSubtask(subtask, availableAgents);
            
            assignments.push({
                subtask,
                agentId: bestAgent.id,
                reason: `Best match for subtask requirements`
            });
        }
        
        return assignments;
    }

    async findBestAgentForSubtask(subtask, availableAgents) {
        // Simple matching based on agent capabilities and subtask description
        // In production, this would use more sophisticated matching algorithms
        
        const agentsArray = Array.from(this.agents.values()).filter(agent => 
            availableAgents.includes(agent.id)
        );

        // Return first available agent for now
        // Enhanced matching would consider agent specialization, past performance, etc.
        return agentsArray[0];
    }

    // Utility methods
    generateAgentId() {
        return `agent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    generateTaskId() {
        return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    // Get agent performance metrics
    getAgentPerformance(agentId) {
        const agent = this.agents.get(agentId);
        if (!agent) return null;

        const completedTasks = this.taskQueue.filter(t => 
            t.agentId === agentId && t.status === 'completed'
        );
        
        const failedTasks = this.taskQueue.filter(t => 
            t.agentId === agentId && t.status === 'failed'
        );

        return {
            totalTasks: completedTasks.length + failedTasks.length,
            successRate: completedTasks.length / (completedTasks.length + failedTasks.length) || 0,
            averageCompletionTime: this.calculateAverageCompletionTime(completedTasks),
            recentActivity: agent.memory.slice(-10)
        };
    }

    calculateAverageCompletionTime(completedTasks) {
        if (completedTasks.length === 0) return 0;
        
        const totalTime = completedTasks.reduce((sum, task) => {
            return sum + (task.completedAt - task.createdAt);
        }, 0);
        
        return totalTime / completedTasks.length;
    }
}

module.exports = SyntheticIntelligence;
