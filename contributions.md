# **Contributions**

This document outlines the contributions of each team member to the **Advanced NLP Assignment 2: End-to-End NLP System Building** at Carnegie Mellon University.

## **Sukriti**
- **Data Collection & Annotation**:
  - Collected **500 questions** covering diverse topics about Pittsburgh and CMU. (With Shambhavi)
  - Filtered and finalized **180** questions based on knowledge base relevance and slight noise inclusion.
  - Ensured dataset diversity with different answer types (**text, number, year, and mixed text-number answers**).
- **Web Scraping & Knowledge Base Preparation**:
  - Extracted data from **Pittsburgh Symphony, Opera, all Sports, and Festivals pages**, and **Cultural Trust events**.
- **Data Processing & Retrieval Setup**:
  - Performed **document chunking** and converted processed data into **VectorDB** for efficient retrieval.
- **Model Development & Output Processing**:
  - Implemented **few-shot learning** techniques for model experimentation.
  - Developed post-processing scripts to clean system outputs and remove irrelevant or extra text.

## **Shambhavi**
- **Data Collection & Baseline Model**:
  - Collected **500 questions** covering diverse topics about Pittsburgh and CMU (With Sukriti)
  - Built a **baseline model** using `google/flan-t5-small` for initial QA system testing.
- **Model Evaluation & Performance Analysis**:
  - Developed the **evaluation script** to compute key metrics: **F1-score, exact match, and BLEU score**.
  - Compared system outputs against reference answers and identified areas for improvement.
- **Final Report Contribution**:
  - Analyzed model results and contributed findings to the final report.

## **Maitri Gada**
- **Data Annotation & Data Scraping**:
  - Scraped information from **Wikipedia, City of Pittsburgh website, Visit Pittsburgh, CMU About page, Drama CMU, Pittsburgh regulations, and events pages**.
  - Annotated the dataset with **Sukriti** using Google Sheets to ensure high-quality reference answers.
- **Model Development & Optimization**:
  - Implemented the **Retrieval-Augmented Generation** pipeline (without fewshot)
  - Worked on **prompt engineering** and conducted ablations on the base model to enhance answer accuracy and informativeness.
- **Final Report & Submission**:
  - Helped compile results and insights for the final report.
  - Assisted in formatting and structuring the final submission.

## **Joint Contributions**
- **Report Writing**:
  - All team members collaborated on different sections of the final report.
- **Final System Testing**:
  - Conducted manual checks on system outputs before submission.
