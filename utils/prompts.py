from langchain.prompts import PromptTemplate


extraction_system_prompt = """
Role: You are an AI expert specializing in extracting structured knowledge from SEC 10-K filings for financial analysis.

Goal: Extract factual Subject-Predicate-Object (SPO) triples from the provided text chunk that are relevant to the company's business operations, financial performance, risks, strategy, and key relationships.

Core Instructions:

1.  Identify Triples: Find relationships where a specific Subject is linked to a specific Object via a clear Predicate (action or relationship).
2.  Relevance: Focus *only* on information valuable for understanding the company's financial health, strategy, market position, risks, and significant business activities. Ignore boilerplate, generic statements, or information not specific to the company's performance or operations.
3.  Specificity:
    *   Subject: Must be a specific named entity (e.g., the company `{company_name}`, a competitor, a product name, a regulation, a specific metric). Use `{company_name}` instead of "the Company", "we", "our".
    *   Predicate: Use a concise, clear verb phrase describing the relationship. **Crucially, use standardized predicates whenever possible.** See the PREFERRED PREDICATES list below. Predicates should typically be 1-3 words.
    *   Object: Must be specific. Include quantities, dates, locations, percentages, or other named entities where available (e.g., "$10.5 billion", "fiscal 2023", "TechCorp", "supply chain disruptions").
4.  Formatting:
    *   Output **ONLY** a valid JSON object.
    *   The JSON object must contain a single key: `"triples"`.
    *   The value of `"triples"` must be a list of JSON objects.
    *   Each inner object must have three keys: `"subject"`, `"predicate"`, `"object"`.
    *   Values for `"subject"` and `"predicate"` should be lowercase strings.
    *   Values for `"object"` should generally retain their original form from the text (especially numbers, dates, proper nouns) but be represented as strings within the JSON.
5.  Quality: Prioritize triples with specific metrics, quantifiable risks, key partnerships, strategic moves, and operational facts. Avoid redundancy.
6.  USER MIGHT BE GIVING TICKER INSTEAD OF COMPANY NAME you should always replace it with the actual company name.


PREFERRED PREDICATES (Use these when applicable, otherwise create a concise alternative):
*   `generated_revenue`, `reported_income`, `has_asset`, `has_liability`, `incurred_expense`
*   `located_in`, `operates_in`, `headquartered_in`
*   `acquired`, `partnered_with`, `subsidiary_of`, `employs`
*   `faces_risk_from`, `mitigates_risk_by`, `dependent_on`
*   `develops_product`, `manufactures_product`, `sells_product`
*   `has_patent`, `filed_lawsuit`, `subject_to_regulation`
*   `effective_date`, `fiscal_year_end`, `reports_metric`


MANDATORY EXTRACTION RULES:

Extract meaningful knowledge graph triples from the following 10-K filing text.

EXTRACTION GUIDELINES:

1. ENTITY UNDERSTANDING:
   - Identify key entities and their natural types
   - Consider the entity's role in the financial context
   - Preserve important attributes and metadata
   - Group related entities appropriately

2. RELATIONSHIP ANALYSIS:
   - Identify meaningful connections between entities
   - Use specific, action-oriented relationships (**Prefer terms from PREFERRED PREDICATES list**)
   - Consider temporal and causal relationships
   - Capture quantitative relationships where relevant

4. CONTEXT PRESERVATION:
   - Note the section and context of each triple
   - Maintain links between related information
   - Preserve temporal and conditional aspects
   - Include relevant qualifiers and conditions

5. RELEVANCE AND QUALITY:
   - Extract ONLY triples that provide meaningful insights for 10-K financial analysis
   - Each triple must contain specific, verifiable information
   - Focus on material information that impacts business value
   - Prioritize quantifiable metrics and relationships

6. FORMATTING REQUIREMENTS:
   - Predicates: Use 1-3 word active verbs/phrases only (**Strongly prefer terms from PREFERRED PREDICATES list**)
   - Case: ALL values must be lowercase
   - Numbers: Preserve exact figures, dates, and percentages
   - Consistency: Use standardized predicate terms across triples (**Adhere to PREFERRED PREDICATES list**)

7. RELATIONSHIP TYPES TO EXTRACT:
   - Financial metrics and performance indicators
   - Risk factors with specific impacts
   - Strategic partnerships and key relationships
   - Market position and competitive advantages
   - Regulatory compliance status
   - Corporate governance structures
   - Operational metrics and KPIs

8. QUALITY VALIDATION:
   - Each triple must be factual and verifiable
   - No generic or boilerplate statements
   - No forward-looking statements without metrics
   - No duplicate or redundant information
   - No vague or ambiguous relationships

Examples of Desired Output (Reflecting Preferred Predicates):

*   Input Text Snippet: "In fiscal 2023, we generated $10.5 billion in revenue, primarily driven by our Cloud Services segment. We face significant competition from TechCorp and Global Solutions Inc."
    Company Name: InnovateCorp
*   Output JSON:
    
    {{
      "triples": [
        {{
          "subject": "innovatecorp",
          "predicate": "generated_revenue",
          "object": "$10.5 billion"
        }},
        {{
          "subject": "innovatecorp revenue generation",
          "predicate": "in_fiscal_year",
          "object": "2023"
        }}, 
        {{
          "subject": "innovatecorp revenue",
          "predicate": "driven_by",
          "object": "Cloud Services segment"
        }},
        {{
          "subject": "innovatecorp",
          "predicate": "faces_risk_from",
          "object": "competition from TechCorp"
        }},
        {{
          "subject": "innovatecorp",
          "predicate": "faces_risk_from",
          "object": "competition from Global Solutions Inc."
        }}
      ]
    }}

*   Input Text Snippet: "A key risk involves potential supply chain disruptions for our 'Widget Pro' product line manufactured in Vietnam."
    Company Name: InnovateCorp
*   Output JSON:
    
    {{
      "triples": [
        {{
          "subject": "innovatecorp",
          "predicate": "faces_risk_from",
          "object": "potential supply chain disruptions"
        }},
        {{
          "subject": "potential supply chain disruptions",
          "predicate": "affects_product",
          "object": "Widget Pro"
        }},
        {{
          "subject": "Widget Pro",
          "predicate": "manufactured_in",
          "object": "Vietnam"
        }}
      ]
    }}
*   Input Text Snippet: "The company entered into a strategic partnership with Alpha Solutions on March 15, 2024, to enhance AI capabilities."
    Company Name: InnovateCorp
*   Output JSON:
    
    {{
      "triples": [
        {{
          "subject": "innovatecorp",
          "predicate": "partnered_with",
          "object": "Alpha Solutions"
        }},
        {{
          "subject": "innovatecorp partnership with alpha solutions",
          "predicate": "effective_date",
          "object": "March 15, 2024"
        }},
        {{
          "subject": "innovatecorp partnership with alpha solutions",
          "predicate": "purpose",
          "object": "enhance AI capabilities"
        }}
      ]
    }}

---

Process the following text chunk:
Text Chunk:
```text
{text_chunk}
Company Name: {company_name}
```


"""
extraction_user_prompt_template = """
Please extract Subject-Predicate-Object (S-P-O) triples from the text below.

MANDATORY EXTRACTION RULES:
Extract meaningful knowledge graph triples from the following 10-K filing text. 

EXTRACTION GUIDELINES:

1. ENTITY UNDERSTANDING:
   - Identify key entities and their natural types
   - Consider the entity's role in the financial context
   - Preserve important attributes and metadata
   - Group related entities appropriately

2. RELATIONSHIP ANALYSIS:
   - Identify meaningful connections between entities
   - Use specific, action-oriented relationships
   - Consider temporal and causal relationships
   - Capture quantitative relationships where relevant

4. CONTEXT PRESERVATION:
   - Note the section and context of each triple
   - Maintain links between related information
   - Preserve temporal and conditional aspects
   - Include relevant qualifiers and conditions

5. RELEVANCE AND QUALITY:
   - Extract ONLY triples that provide meaningful insights for 10-K financial analysis
   - Each triple must contain specific, verifiable information
   - Focus on material information that impacts business value
   - Prioritize quantifiable metrics and relationships

6. FORMATTING REQUIREMENTS:
   - Predicates: Use 1-2 word active verbs/phrases only
   - Case: ALL values must be lowercase
   - Numbers: Preserve exact figures, dates, and percentages
   - Names: Use full entity names (e.g., 'apple inc.' not 'apple')
   - Consistency: Use standardized predicate terms across triples

7. RELATIONSHIP TYPES TO EXTRACT:
   - Financial metrics and performance indicators
   - Risk factors with specific impacts
   - Strategic partnerships and key relationships
   - Market position and competitive advantages
   - Regulatory compliance status
   - Corporate governance structures
   - Operational metrics and KPIs

8. QUALITY VALIDATION:
   - Each triple must be factual and verifiable
   - No generic or boilerplate statements
   - No forward-looking statements without metrics
   - No duplicate or redundant information
   - No vague or ambiguous relationships

Text to Process:
```text
{text_chunk}
```
Company Ticker: {company_name}
NOTE : ALWAYS REPLACE any generic entity names with the actual company name. 
OUTPUT FORMAT:
Return ONLY a JSON object with highest quality triples:
{{
    "triples": [
        {{
            "subject": "specific entity (lowercase)",
            "predicate": "active verb/phrase (1-2 words)",
            "object": "specific measurable target (lowercase)"
        }}
    ]
}}
"""

schema_definition_prompt_template = """
You will be given a piece of text and a list of relational triples in the format of [Subject, Relation, Object] extracted from the text. For each relation present in the triples, your task is to write a description to express the meaning of the relation.

GUIDELINES FOR RELATION DEFINITIONS:
1. Each definition should be clear, concise, and general enough to apply to other entities
2. Focus on the semantic meaning of the relation in a business/financial context
3. Pay attention to the direction (subject to object) of the relation
4. Ensure consistency across related definitions
5. Use formal, professional language appropriate for financial documentation

Here are some examples:
{few_shot_examples}

Now please extract relation descriptions given the following:
Text: {text}
Triples: {triples}
Relations: {relations}

OUTPUT FORMAT:
Return definitions in a structured format, one per line:
relation_name: definition
"""


qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert financial analyst. Use the following context from a knowledge graph to answer the user's question.

Context:
{context}

Question:
{question}

Answer in a concise, factual manner:
"""
)

entity_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Extract all relevant entities from the following question for use in a knowledge graph query.

Question: {question}

Entities (comma-separated):
"""
)