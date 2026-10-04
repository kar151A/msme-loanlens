# MSME LoanLens

> Explainable AI-powered MSME loan decision assistant using Machine Learning, Counterfactual Reasoning, RAG and Vector Search.

---

## Overview

MSME LoanLens is an AI-powered loan decision assistant designed to make MSME lending decisions more transparent and actionable.

Instead of only predicting whether a loan may be approved or rejected, LoanLens also explains the likely reasons behind the decision, retrieves similar historical loan applications, and generates counterfactual recommendations such as:

> "If the business reduces debt or improves its credit profile, the predicted approval probability may increase."

The project combines traditional machine learning with modern Generative AI and retrieval-based techniques.

---

## Problem Statement

Many MSME borrowers receive loan rejections with little or no explanation.

This creates three major problems:

- Borrowers do not know why their application was rejected.
- They do not know what changes could improve approval chances.
- Lending decisions can behave like black-box systems.

LoanLens aims to solve this by providing explainable and actionable decision support.

---

## Core Features

### 1. Loan Approval Prediction
Uses a Random Forest classifier to estimate whether an MSME loan application is likely to be approved or rejected.

### 2. Explainable Decision Support
Provides interpretable information using financial features such as:

- CIBIL score
- Business vintage
- Annual revenue
- Existing debt
- Loan amount
- Debt-to-revenue ratio
- Loan-to-revenue ratio

### 3. Counterfactual Reasoning
Generates "what-if" scenarios such as:

- What if CIBIL increases?
- What if debt is reduced?
- What if business vintage improves?
- What if the requested loan amount decreases?

### 4. Similar Historical Case Retrieval
Uses vector embeddings and FAISS to retrieve similar historical loan applications.

### 5. Retrieval-Augmented Generation
Combines retrieved cases, ML predictions, and counterfactuals with an LLM to generate human-readable explanations.

### 6. Interactive Web Application
The final application will allow users to enter their loan details and receive:

- Predicted decision
- Approval probability
- Key reasons
- Similar historical cases
- Counterfactual recommendations
- AI-generated explanation

---

## System Architecture

```text
                         USER
                          |
                          v
                  Web Application
                  Streamlit / React
                          |
                          v
                       FastAPI
                          |
          --------------------------------
          |              |               |
          v              v               v
   ML Prediction     FAISS Retrieval   Counterfactual
   Random Forest      Embeddings         Engine
          |              |               |
          ---------------|---------------
                          |
                          v
                     RAG Pipeline
                          |
                          v
                        LLM
                          |
                          v
                Explainable Loan Report
