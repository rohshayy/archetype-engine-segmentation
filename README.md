# Archetype-Engine: High-Dimensional Behavioral Segmentation

## **Project Overview**
Archetype-Engine is a sophisticated unsupervised learning system designed to transform noisy, high-dimensional raw data into actionable business intelligence. It solves the **"Curse of Dimensionality"** by utilizing a dual-stage mathematical pipeline to identify latent patterns in user behavior.

## **Core Methodology**

### **Phase I: Feature Orthogonalization (PCA)**
The engine applies **Principal Component Analysis (PCA)** to the standardized dataset.
* **The Math:** By performing **Singular Value Decomposition (SVD)**, the engine identifies the eigenvectors that capture the maximum variance.
* **The Result:** 50 correlated behavioral metrics are compressed into 3 independent (orthogonal) Principal Components. This removes redundancy and ensures the subsequent clustering is based on variance "Signal" rather than stochastic "Noise."

### **Phase II: Automated Persona Discovery (K-Means)**
Unlike standard clustering scripts, Archetype-Engine does not rely on human guesswork to determine the number of groups ($k$).
* **The Math:** The system utilizes **Silhouette Analysis**, calculating the ratio of inter-cluster separation to intra-cluster density ($s = \frac{b-a}{max(a,b)}$).
* **The Result:** The pipeline mathematically selects the optimal number of "Archetypes" by maximizing the Silhouette Coefficient.

## **Explainable AI (XAI): The "Glass Box" Approach**
To provide value to non-technical stakeholders, the engine includes a **Feature Loading Analysis** module.
* **The Logic:** It extracts the absolute weights (loadings) from the primary Principal Component to identify which original metrics were the primary drivers of the variance.
* **The Impact:** This provides transparency, explaining *why* a specific segment exists based on its dominant behavioral drivers.

## **Visual Intelligence Outputs**
1.  **3D Latent Space Scatter:** A geometric visualization of archetype separation in reduced-dimensional space.
2.  **Feature Driver Analysis:** A ranking of the top influence-metrics to provide model interpretability.

## **Technical Stack**
* **Language:** Python
* **Libraries:** Scikit-Learn (PCA, KMeans), NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
