# AI & ML Projects

A collection of multi-omics, machine learning and AI exercises covering a range of techniques, datasets, and deployment patterns.

## Projects

| Project | Description | Methods |
|---|---|---|
| [Home Price Prediction](home_price_prediction/) | Predict residential sale prices from 80 housing features | Linear Regression, XGBoost, Gradio |
| [First LLM](first_llm/) | Intro to LLM inference with LangChain and Groq | Llama 3.1, GPT-o, LangChain |
| [Compound Classifier](compound_classifier/) | Multi-model classification of compounds from molecular features | Elastic Net, Random Forest, XGBoost, SVM, Neural Net, Ensemble |
| [TensorFlow Banknote Authentication](deep_learning_tensor_exercise/) | Binary classification of authentic vs. fake banknotes | TensorFlow DNN, Random Forest |
| [Chatbot with Memory](ChatBot_with_Memory/) | Conversational chatbot with persistent in-session memory and Gradio UI | LangChain, Groq, Gradio |
| [Hauck-Donner Effect](Hauck_Donner_Effect/) | Simulation of the Wald test pathology in logistic regression association studies | Base R, GLM, LRT, Permutation test |
| [SMILES to Pathway](smiles_to_pathway/) | Biological pathway enrichment from chemical structure input | RDKit, PyTorch, GSEApy, KEGG |
| [Clinical Trial Analysis](clinical_trial_test_data/) | Longitudinal biomarker analysis across treatment groups and time points | lme4, emmeans, ggplot2 |
| [scRNA-seq Pipeline](tsc_rnaseq/) | Single-cell RNA-seq simulation and full analysis pipeline | scanpy, Harmony, Leiden, Wilcoxon DE |
| [Omics Foundation Model Evaluation](omics_fm_eval/) | Benchmark suite comparing single-cell foundation models against classical baselines, with an LLM-as-evaluator layer | Geneformer, PCA, scanpy, LLM-as-judge |
| [RAG over Drug Literature](rag_drug_lit/) | Retrieval-augmented Q&A over real PubMed abstracts with cited answers, a hallucination guard, and a two-tier evaluation | TF-IDF retrieval, Claude, DeepEval (faithfulness/relevancy), PubMed + PDF ingestion |
| [Compound Indication Classifier](compound_indication_classifier/) | Predict a compound's therapeutic indication from its SMILES structure | RDKit descriptors vs Morgan fingerprints, Random Forest, 10-fold CV |
| [CNN Skin-Lesion Classifier](cnn_skin_lesion/) | Convolutional network on DermaMNIST with a proper training recipe and imbalance-aware metrics | PyTorch CNN & ResNet-18, LR schedule, AUC / balanced-acc / macro-F1 |
| [AlphaMissense vs ClinVar](alphamissense_clinvar_benchmark/) | Benchmark AlphaMissense pathogenicity scores against ClinVar labels; surfaces a critical missed mutation | MyVariant.info, ROC-AUC / PR-AUC, ClinVar / dbNSFP |
| [Perturb-seq Analysis](perturbseq_analysis/) | CRISPR screen analysis separating true knockouts from escapers and ranking perturbation strength | pertpy, Mixscape, E-distance, scanpy |
| [scVI Integration & Cell Typing](scvi_integration/) | Batch-integration benchmark (PCA vs Harmony vs scVI) plus scANVI cell-type annotation via label transfer | scvi-tools (scVI/scANVI), Harmony, scanpy |

---

*More projects coming soon.*
