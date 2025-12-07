---
license: cdla-sharing-1.0
task_categories:
- question-answering
- table-question-answering
language:
- en
tags:
- question-answering
- llm
- chatbot
- hospitality
- conversational-ai
- generative-ai
- natural-language-understanding
- fine-tuning
pretty_name: >-
  Bitext - Hospitality Tagged Training Dataset for LLM-based Virtual Assistants
size_categories:
- 10K<n<100K
---
# Bitext - Hospitality Tagged Training Dataset for LLM-based Virtual Assistants

## Overview

This hybrid synthetic dataset is designed to be used to fine-tune Large Language Models such as GPT, Mistral and OpenELM, and has been generated using our NLP/NLG technology and our automated Data Labeling (DAL) tools. The goal is to demonstrate how Verticalization/Domain Adaptation for the [hospitality] sector can be easily achieved using our two-step approach to LLM Fine-Tuning. An overview of this approach can be found at: [From General-Purpose LLMs to Verticalized Enterprise Models](https://www.bitext.com/blog/general-purpose-models-verticalized-enterprise-genai/)

The dataset has the following specifications:

- Use Case: Intent Detection
- Vertical: Hospitality
- 25 intents assigned to 16 categories
- 25000 question/answer pairs, with approximately 1000 per intent
- 40 entity/slot types
- 10 different types of language generation tags

The categories and intents are derived from Bitext's extensive experience across various industry-specific datasets, ensuring the relevance and applicability across diverse contexts.

## Dataset Token Count

The dataset contains a total of 3.15 million tokens across 'instruction' and 'response' columns. This extensive corpus is crucial for training sophisticated LLMs that can perform a variety of functions including conversational AI, question answering, and virtual assistant tasks in the hospitality domain.

## Fields of the Dataset

Each entry in the dataset comprises the following fields:

- tags
- instruction: a user request from the Hospitality domain
- category: the high-level semantic category for the intent
- intent: the specific intent corresponding to the user instruction
- response: an example of an expected response from the virtual assistant

## Categories and Intents

The dataset covers a wide range of hospitality-related categories and intents, which are:

- **BILLING**: invoices
- **CANCELLATION_FEES**: cancellation_fees
- **CHECK_IN**: check_in
- **CHECK_OUT**: check_out
- **CONTACT**: customer_service, human_agent
- **EVENT**: host_event
- **FEEDBACK**: file_complaint, leave_review
- **HOTEL**: book_hotel, cancel_hotel_reservation, change_hotel_reservation, check_hotel_facilities, check_hotel_offers, check_hotel_prices, check_hotel_reservation, search_hotel
- **LUGGAGE**: store_luggage
- **MENU**: check_menu
- **NIGHT**: add_night
- **PARKING_SPACE**: book_parking_space
- **PETS**: bring_pets
- **POINTS**: redeem_points
- **REFUND**: get_refund
- **SHUTTLE_SERVICE**: shuttle_service

## Entities

The entities covered by the dataset include:

- **{{WEBSITE_URL}}**, common with most intents.
- **{{SPEAK_WITH_HUMAN_AGENT_OPTION}}**, featured in intents like human_agent.
- **{{CHECK_OUT_OPTION}}**, associated with intents like check_out.
- **{{CHARGE_TYPE}}**, relevant to intents such as cancellation_fees.
- **{{DESTINATION_CITY}}**, featured in intents like book_hotel, change_hotel_reservation.

This comprehensive list of entities ensures that the dataset is well-equipped to train models that are highly adept at understanding and processing a wide range of hospitality-related queries and tasks.

## Language Generation Tags

The dataset includes tags indicative of various language variations and styles adapted for Hospitality, enhancing the robustness and versatility of models trained on this data. These tags categorize the utterances into different registers such as colloquial, formal, or containing specific hospitality jargon, ensuring that the trained models can understand and generate a range of conversational styles appropriate for different customer interactions in the hospitality sector.

## Language Generation Tags

The dataset includes tags that reflect various language variations and styles, crucial for creating adaptable and responsive conversational AI models within the hospitality sector. These tags help in understanding and generating appropriate responses based on the linguistic context and user interaction style.

### Tags for Lexical variation

- **M - Morphological variation**: Adjusts for inflectional and derivational forms.
  - Example: "is my account active", "is my account activated"
- **L - Semantic variations**: Handles synonyms, use of hyphens, and compounding.
  - Example: “what's my balance date", “what's my billing date”

### Tags for Syntactic structure variation

- **B - Basic syntactic structure**: Simple, direct commands or statements.
  - Example: "activate my card", "I need to check my balance"
- **I - Interrogative structure**: Structuring sentences in the form of questions.
  - Example: “can you show my balance?”, “how do I transfer money?”
- **C - Coordinated syntactic structure**: Complex sentences coordinating multiple ideas or tasks.
  - Example: “I want to transfer money and check my balance, what should I do?”
- **N - Negation**: Expressing denial or contradiction.
  - Example: "I do not wish to proceed with this transaction, how can I stop it?"

### Tags for language register variations

- **P - Politeness variation**: Polite forms often used in customer service.
  - Example: “could you please help me check my account balance?”
- **Q - Colloquial variation**: Informal language that might be used in casual customer interactions.
  - Example: "can u tell me my balance?"
- **W - Offensive language**: Handling potentially offensive language which might occasionally appear in frustrated customer interactions.
  - Example: “I’m upset with these charges, this is ridiculous!”

### Tags for stylistic variations

- **K - Keyword mode**: Responses focused on keywords.
  - Example: "balance check", "account status"
- **E - Use of abbreviations**: Common abbreviations.
  - Example: “acct for account”, “trans for transaction”
- **Z - Errors and Typos**: Includes common misspellings or typographical errors found in customer inputs.
  - Example: “how can I chek my balance”

### Other tags not in use in this Dataset

- **D - Indirect speech**: Expressing commands or requests indirectly.
  - Example: “I was wondering if you could show me my last transaction.”
- **G - Regional variations**: Adjustments for regional language differences.
  - Example: American vs British English: "checking account" vs "current account"
- **R - Respect structures - Language-dependent variations**: Formality levels appropriate in different languages.
  - Example: Using “vous” in French for formal addressing instead of “tu.”
- **Y - Code switching**: Switching between languages or dialects within the same conversation.
  - Example: “Can you help me with my cuenta, please?”

These tags not only aid in training models for a wide range of customer interactions but also ensure that the models are culturally and linguistically sensitive, enhancing the customer experience in hospitality environments.

## License

The `Bitext-hospitality-llm-chatbot-training-dataset` is released under the **Community Data License Agreement (CDLA) Sharing 1.0**. This license facilitates broad sharing and collaboration while ensuring that the freedom to use, share, modify, and utilize the data remains intact for all users.

### Key Aspects of CDLA-Sharing 1.0

- **Attribution and ShareAlike**: Users must attribute the dataset and continue to share derivatives under the same license.
- **Non-Exclusivity**: The license is non-exclusive, allowing multiple users to utilize the data simultaneously.
- **Irrevocability**: Except in cases of material non-compliance, rights under this license are irrevocable.
- **No Warranty**: The dataset is provided without warranties regarding its accuracy, completeness, or fitness for a particular purpose.
- **Limitation of Liability**: Both users and data providers limit their liability for damages arising from the use of the dataset.

### Usage Under CDLA-Sharing 1.0

By using the `Bitext-hospitality-llm-chatbot-training-dataset`, you agree to adhere to the terms set forth in the CDLA-Sharing 1.0. It is essential to ensure that any publications or distributions of the data, or derivatives thereof, maintain attribution to the original data providers and are distributed under the same or compatible terms of this agreement.

For a detailed understanding of the license, refer to the [official CDLA-Sharing 1.0 documentation](https://cdla.dev/sharing-1-0/).

This license supports the open sharing and collaborative improvement of datasets within the AI and data science community, making it particularly suited for projects aimed at developing and enhancing AI technologies in the hospitality sector.

---

(c) Bitext Innovations, 2024
