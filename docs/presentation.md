# Comparison of XAI Methods for Multimodal Analysis

---

## Slide 1: Project Overview
- **Objective**: Compare XAI methods (GradCAM, SHAP, LIME) on image and text datasets
- **Datasets**:
  - Image: Tumor detection (benign/malignant)
  - Text: Tweet sentiment analysis (negative/neutral/positive)
- **Methods**:
  - GradCAM: Visual explanations for CNN-based image models
  - SHAP: Feature importance for both image and text
  - LIME: Local interpretable model-agnostic explanations

---

## Slide 2: Technical Implementation - GradCAM
- **Key Features**:
  - Captum's LayerGradCam implementation
  - Automatic target layer selection
  - Heatmap generation with interpolation and normalization
- **Visualization**:
  ![GradCAM Example](results/xai_results_20250110_181128_all_mode_and_samples/explanations_0.png)

---

## Slide 3: Technical Implementation - SHAP
- **Key Features**:
  - Image: Partition masker with blurring
  - Text: Word-level importance by prediction difference
  - Normalized feature importance scores
- **Visualization**:
  ![SHAP Example](results/xai_results_20250110_181128_all_mode_and_samples/explanations_100.png)

---

## Slide 4: Technical Implementation - LIME
- **Key Features**:
  - Image: Superpixel segmentation and visualization
  - Text: Word importance scores
  - Model-agnostic local explanations
- **Visualization**:
  ![LIME Example](results/xai_results_20250110_181128_all_mode_and_samples/explanations_200.png)

---

## Slide 5: Comparative Analysis
- **Evaluation Metrics**:
  - Faithfulness: How well explanations match model behavior
  - Stability: Consistency across similar inputs
  - Interpretability: Human understanding of explanations
- **Findings**:
  - GradCAM best for CNN-based image models
  - SHAP provides global feature importance
  - LIME offers local interpretability

---

## Slide 6: Conclusion & Future Work
- **Key Takeaways**:
  - Different XAI methods suit different use cases
  - Multimodal analysis requires modality-specific approaches
  - Trade-offs between accuracy and interpretability
- **Future Directions**:
  - Unified multimodal explanation framework
  - Quantitative evaluation metrics
  - Real-time explanation generation
