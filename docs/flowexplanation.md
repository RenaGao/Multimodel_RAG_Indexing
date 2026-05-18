Round 1:

user proxy agent send a user query -> dialogue history agent use the user query to do rag search (should be empty for 1st round) -> recommendation agent use the user query to do rag search -> reasoning agent explain based on dialogue history agent + rag search result and recommendation agent + rag search result

Round 2:

user proxy agent send a next user query -> dialogue history agent use the user queyr to do rag search (now the rag search result should combine the dialogue history raw data and rag search result based on the user query), then let dialogue history agent use conversation history as reference and focus more on rag search result, then generate some user labels -> recommendation agent use user query and (may be not) user labels to do rag search then give recommendations -> reasoning agent still explain based on this round's dialogue history agent and recommendation agent + tools' result.

rag search for dialogue history agent: 

input: user query (maybe) topk

output rag search use user query embedding and retrieve related information from chat history, and conversation history raw data include tool's result

rag search for recommendation agent: 

input: user query, (maybe) user labels, (maybe) topk

output: product data





Updated: 

dialogue history agent - rag search: topk 人定 - needs experiment
{
    {
        raw dialogue history data ? - retrieve most relevent dialogue rounds data 
    },
    {
        rag retrieve data - agent focus more on this part, use history raw data as reference
    }
}

- agent output summary



recommend agent - rag search: 


input user query (all previous user queries) only, first step retrieve product -> 

preference from dialogus history or history review 

input user query, retrieve -> recommendation
output data 

user query -> function rag search product + function retrieve user's all previous review history -> combine result then return
recommendation agent uses dialogue history agent's result (user preference (summary))-> rerank (fliter) products

summary retrieve function ()

{
    {},
    {},
    {}
}

reason agent - generate reason based on dialogue history agent + recommendation agent + tool's result + this round



