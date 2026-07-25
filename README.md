# AI Customer Segmentation & Personalization Agent for Retail Banking

## Problem Statement
Retail banks face low engagement and suboptimal product adoption due to broad, generic marketing strategies. This solution delivers an AI-powered agentic system that ingests customer transaction data, performs automated EDA, segments customers based on financial behavior, generates explainable personas, and recommends targeted cross-sell financial products.

## Dataset Information
- **Source:** Publicly sourced open financial dataset (`bank_transactions.csv` / `CC GENERAL.csv`).
- **Features Used:** Account Balance (`CustAccountBalance`), Transaction Amount (`TransactionAmount`), Transaction Frequency, and Recency.

## Architecture & Agent Design
1. **EDA Tool:** Computes missing values, summary statistics, and correlations dynamically.
2. **Feature Engineering Tool:** Aggregates transactional data into customer-level metrics (`avg_balance`, `transaction_frequency`, `avg_transaction_size`).
3. **Segmentation Tool:** Implements Rule-based & K-Means clustering to classify customers into Priority, Regular, and Dormant groups.
4. **Explainability & Recommendation Tool:** Derives segment rules and links customer personas to tailored banking products.
5. **Gemini Orchestrator:** Uses Google Gemini API with Function Calling to process user natural language queries and trigger appropriate analytical tools.

## Setup & Running Instructions
1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd retail-banking-agent