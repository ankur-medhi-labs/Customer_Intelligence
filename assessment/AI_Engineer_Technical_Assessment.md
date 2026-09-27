Below is the technical assessment and synthetic chat dataset.

The assessment covers a few connected AI use cases:

Customer interaction intelligence - take an informal customer chat and generate a structured formal summary, sentiment, customer temperature, intent, escalation indication, and recommended next action.
Thematic and emerging issue analysis - identify recurring customer themes and detect new or emerging issues from the supplied conversations.
Contextual Q&A with RAG - build a small RAG-based solution using AWS Bedrock. You can use synthetic data or a public source such as www.metricon.com.au for the knowledge base.
We’re mainly interested in your technical approach, working implementation, and how you handle grounding, consistency and evaluation. The UI can be kept simple, as the focus is on the AI solution itself

# AI Engineer - Technical Assessment

**Expected effort:** 4-5 hours\
**Focus:** working code and technical decisions

You have been provided with a synthetic customer chat dataset containing
informal customer conversations and example resolution notes. The aim of
this exercise is to build a small working AI solution using the dataset
and demonstrate how you approach an applied AI problem.

This is not expected to be a production ready application. We are more
interested in the quality of the implementation, the decisions you make,
and how you explain those decisions during the interview.

## 1. Customer Interaction Analysis

Build a component that accepts a customer conversation and returns a
structured result.

The result should include, at minimum:

- Formal summary
- Primary topic
- Customer intent
- Sentiment
- Customer temperature
- Temperature score
- Whether escalation is required
- Recommended next action

The output should be consistent and suitable for use by another
application. It should not introduce facts that are not supported by the
conversation or by another source used by your solution.

## 2. Customer Temperature

Implement a customer temperature measure that helps identify the level
of concern or escalation risk in an interaction.

You may define the scale and scoring approach you believe is
appropriate. Be prepared to explain during the interview what the score
represents, what information contributes to it, and how you would
determine whether it is reliable.

## 3. Theme and Trend Analysis

Use the supplied conversations to identify recurring customer themes and
produce a useful summary of what customers are contacting the business
about.

The solution should also be able to identify a new or emerging issue
that was not already obvious from the existing set of conversations. You
may add a small number of synthetic conversations to demonstrate this
behaviour if required.

The output should make it possible to understand, at a minimum:

- Common themes or issues
- How often each theme appears
- The sentiment or customer temperature associated with those themes
- Any newly emerging issue identified by the solution

## 4. RAG Question and Answer Capability

Build a small retrieval augmented question and answer capability using a
knowledge source of your choice.

This may be a small set of public information, FAQs, policies, product
information, or other suitable content.

A user should be able to ask a question and receive an answer based on
the available source material.

The response should include a reference to the source used. If the
available information does not support an answer, the system should
handle that case without inventing information.

## 5. Evaluation

Include a practical way to evaluate the solution. This should be
implemented as part of the code rather than as a long written report.

The evaluation should cover enough cases to show how you check that the
main behaviours are working, including both expected cases and failure
or unknown cases.

## Technical Requirements

- Python should be used for the main implementation.
- AWS Bedrock is preferred for generative AI components.
- You may use other supporting libraries or services where
  appropriate.
- A simple CLI, API, notebook, or basic user interface is sufficient.
- Front-end design is not part of the assessment.
- The solution should be runnable by another engineer using the
  instructions you provide.

## What to Submit

- Source code
- A short README with setup and run instructions
- Dependency information
- An example environment/configuration file with no secrets
- Any test or evaluation code used to check the solution
