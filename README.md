### Credit Scoring Business Understanding

### 1. Basel II Accord and the Need for Interpretability

The Basel II Capital Accord emphasizes three pillars: minimum capital requirements, supervisory review, and market discipline. In particular, it requires that institutions adopting internal ratings-based (IRB) approaches must demonstrate robust, transparent, and interpretable models for assessing credit risk. This means that black-box models are discouraged in favor of models that can be explained and audited. For Bati Bank, this underscores the importance of building a credit scoring system that is not only accurate but also transparent to regulators, stakeholders, and customers. Models must be well-documented, reproducible, and interpretable to ensure that risk assessments align with regulatory standards.

### 2. Importance of a Proxy Variable for Default

In this project, we lack a direct label indicating whether a customer defaulted on their payments. Hence, we must define a proxy variable based on observed behaviors such as delayed payments, refund requests, fraud flags, or inactivity after receiving credit.

The creation of a proxy is crucial because:

It provides a target variable to train predictive models.

It allows us to simulate and model risk in the absence of direct loan outcomes.

However, relying on proxies introduces potential business risks:

Misclassification: We might incorrectly label a reliable customer as high-risk, leading to missed business opportunities.

Regulatory scrutiny: If the proxy does not accurately represent true default behavior, it may not meet the standards expected under Basel II.

Model bias: Using flawed proxies can propagate biases in the scoring model, leading to systemic discrimination or poor generalization.

### 3. Model Trade-Offs in a Regulated Context

Simple Models (e.g., Logistic Regression with Weight of Evidence)

Pros:

Highly interpretable

Easier to audit and validate

Faster to train and deploy

Accepted by regulators

Cons:

Limited ability to capture non-linear relationships

Lower predictive performance in complex datasets

Complex Models (e.g., Gradient Boosting Machines)

Pros:

Higher accuracy and ability to handle complex feature interactions

Better performance with large feature sets

Cons:

Low interpretability (black-box behavior)

Difficult to validate and explain to regulators

Greater risk of overfitting and bias

 Conclusion: In regulated financial contexts like Bati Bank, it is often preferable to begin with a simpler model for baseline evaluation and compliance. More complex models can be introduced later with proper interpretability layers (e.g., SHAP, LIME) and rigorous documentation to balance performance with transparency.

