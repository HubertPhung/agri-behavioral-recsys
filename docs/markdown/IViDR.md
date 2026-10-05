# Mitigating Dual Latent Confounding Biases in Recommender Systems

**Jianfeng Deng**  
Guangxi University, Nanning, China  
`jianfeng_web@163.com`

**Qingfeng Chen**\*  
Guangxi University, Nanning, China  
`qingfeng@gxu.edu.cn`

**Debo Cheng**\*  
University of South Australia, Adelaide, Australia  
`debo.cheng@unisa.edu.au`

**Jiuyong Li**  
University of South Australia, Adelaide, Australia  
`jiuyong.li@unisa.edu.au`

**Lin Liu**  
University of South Australia, Adelaide, Australia  
`liu.lin@unisa.edu.au`

**Xiaojing Du**  
University of South Australia, Adelaide, Australia  
`xiaojing.du@mymail.unisa.edu.au`

\* Corresponding author.

---

## Abstract

Recommender systems are extensively utilised across various areas to predict user preferences for personalised experiences and enhanced user engagement and satisfaction. Traditional recommender systems, however, are complicated by confounding bias, particularly in the presence of latent confounders that affect both item exposure and user feedback. Existing debiasing methods often fail to capture the complex interactions caused by latent confounders in interaction data, especially when dual latent confounders affect both the user and item sides. To address this, we propose a novel debiasing method that jointly integrates the Instrumental Variables (IV) approach and identifiable Variational Auto-Encoder (iVAE) for Debiased representation learning in Recommendation systems, referred to as **IViDR**. Specifically, IViDR leverages the embeddings of user features as IVs to address confounding bias caused by latent confounders between items and user feedback, and reconstructs the embedding of items to obtain debiased interaction data. Moreover, IViDR employs an Identifiable Variational Auto-Encoder (iVAE) to infer identifiable representations of latent confounders between item exposure and user feedback from both the original and debiased interaction data. Additionally, we provide theoretical analyses of the soundness of using IV and the identifiability of the latent representations. Extensive experiments on both synthetic and real-world datasets demonstrate that IViDR outperforms state-of-the-art models in reducing bias and providing reliable recommendations.

**Keywords:** Recommender systems, Instrumental Variables, Latent Confounders, Debiasing

---

## 1 Introduction

Recommender systems are widely employed to offer personalised suggestions across various domains, including video streaming [10], e-commerce [42], and online search [34]. These systems have evolved significantly with the advances in collaborative filtering techniques (e.g., Matrix Factorisation (MF) [18]), deep learning algorithms (e.g., Deep Factorisation Machine (DeepFM) [11], Deep Cross Network (DCN) [31], Deep Interest Network (DIN) [41]), and graph neural networks (GNNs) (e.g., LightGCN [13]). Despite these innovations, many traditional recommendation algorithms rely heavily on statistical associations, which can introduce estimation biases such as popularity bias [1], selection bias [20, 21], and conformity bias [40]. To address these challenges, some causality-based recommender systems have been developed and successfully deployed in real-world applications [25].

Causal recommender systems address estimation bias by incorporating causal inference techniques. Previous studies have employed classical causal inference methods [24, 26], such as back-door adjustment [24] and inverse propensity reweighting (IPW) [27] to mitigate specific biases. For instance, IPS [27] leverages IPW to address selection bias [20], while D2Q [36] applies back-door adjustments to counter video duration bias, which typically skews recommendations towards longer videos. Additionally, the Popularity De-biasing Algorithm (PDA) [39] specifically uses causal interventions to eliminate popularity bias and improve recommendation accuracy. However, these approaches generally overlook latent confounders, which can still significantly impact recommendation accuracy and fairness [24].

Recent developments in causal recommendation methods have focused on accounting for latent confounders [27, 37], aiming to enhance the robustness and unbiasedness of estimations by modeling and adjusting latent confounders that influence both user behaviour and item exposure. Latent confounders, which are unobservable by nature, introduce substantial estimation bias in recommender systems [5, 6, 24]. For example, high-quality products are usually priced higher and tend to receive more positive ratings from users, leading to a false correlation between high prices and positive ratings. As a result, recommender systems may tend to recommend high-priced products, but high-priced products may be disliked by users. Previous works, such as Deconfounder [32], approximate latent confounders using historical user data to mitigate their effects. The iDCF [37] method employs proximal causal inference techniques to address the impact of latent confounders. However, these methods may fail to infer latent confounders directly from interaction data, as some latent confounders may lack proxy information within the data. As a result, the inference of latent confounders from such data can potentially be inaccurate.

Instrumental variables (IVs) have been used to decompose input vectors within recommendation models to mitigate the effects of latent confounders [4, 5, 12, 30], as demonstrated in methods like IV4Rec [28] and IV4Rec+ [29]. While these approaches can be effective for debiasing, they need to modify the input vectors using representation decomposition, which does not have an identification guarantee for the learned representations. Consequently, different learned representations can yield varying performance outcomes, making it difficult to ascertain whether improvements in accuracy are truly due to the mitigation of latent confounders or simply variations in the input data representations [15]. Furthermore, the decomposed representations used in IV4Rec [28] and IV4Rec+ [29] primarily address latent confounders between items and user feedback, while overlooking the latent confounders between item exposure and user feedback.

Recommender systems deal with complex user-item historical data, where latent confounders can arise from different sources. Broadly, latent confounders fall into two categories: those that can be inferred through proxy variables and those without reliable proxy variables. For example, a user's socioeconomic status can be estimated by the average price of products they have purchased, whereas a user's mood when providing ratings lacks a clear proxy variable. Existing methods typically address only one type of latent confounder. For example, iDCF [37] primarily focuses on latent confounders with proxy variables, while methods like IV4Rec [28] and IV4Rec+ [29] handle latent confounders without proxy variables but may overlook those with proxy variables, such as latent confounders between item exposure and user feedback. Since IV4Rec [28] does not have an identification guarantee, simply combining IV4Rec and iDCF to address both types of latent confounders still fails to achieve identifiability on learned representation. To the best of our knowledge, no practical solution currently exists that addresses both types of latent confounders (i.e., dual latent confounding biases) in recommender systems.

To address this challenge, we propose a novel debiasing method (**IViDR**) that jointly integrates the Instrumental Variables (IV) approach and identifiable VAE for Debiased representation learning in Recommendation systems to mitigate dual latent confounding biases. Specifically, IViDR first utilises user feature embeddings as IVs to reconstruct the treatment variables and generate debiased interaction data for addressing the effects of between items and user feedback. Subsequently, IViDR employs an iVAE [15] to infer identifiable latent representations from a combination of additional observable proxy variables (e.g., user's consumption level), interaction data, and debiased interaction data. Finally, IViDR adjusts for the inferred latent representations to mitigate confounding bias between item exposure and user feedback, resulting in a debiased recommendation system. In summary, our main contributions are listed as follows:
- We propose a novel debiasing method, IViDR, for mitigating dual latent confounding biases, i.e., the confounding biases caused by latent confounders between items and user feedback, as well as between item exposure and user feedback.
- We provide a theoretical analysis of the identification of the learned latent representations in the IViDR, along with a discussion on the validity of using IVs.
- Extensive experiments on both synthetic and real-world datasets demonstrate that IViDR outperforms state-of-the-art recommendation methods in reducing bias and improving recommendation accuracy.

---

## 2 Related Work

In this section, we review methods for recommender systems, which can broadly be categorized into two groups: traditional correlation-based systems and causality-based systems.

**Traditional Recommendation Algorithms.** Traditional algorithms include collaborative filtering-based approaches (e.g., Matrix Factorisation (MF) [18]), neural network-based recommendation algorithms (e.g., Deep Factorisation Machine (DeepFM) [11], Deep Cross Network (DCN) [31], Deep Interest Network (DIN) [41]), and graph neural network-based algorithms (e.g., LightGCN [13]). However, these methods are susceptible to introducing biases into recommendation outcomes. Instead of delving deeply into traditional algorithms, we direct our attention to causality-based recommender systems. For more comprehensive reviews of traditional approaches, please refer to [2, 38].

**Causal Recommendation Algorithms for Addressing Specific Biases.** With advancements in causal inference techniques [24, 26], causality-based recommendation algorithms have emerged. Several studies have adopted classical causal inference methods, such as back-door adjustment [24] or IPW [27], to mitigate specific biases. For instance, IPS [27] uses IPW to address selection bias [20], while D2Q [36] employs back-door adjustment to reduce video duration bias, which often leads recommender systems to favour longer videos. The Popularity De-biasing Algorithm (PDA) [39] tackles popularity bias to enhance recommendation accuracy. Despite their promise, these algorithms typically overlook the influence of latent confounders, which can undermine the accuracy of classical causal inference methods [5, 7].

To address the challenges posed by latent confounders, recent research has proposed several debiasing methods. The Deconfounder [32] seeks to approximate latent confounders using historical user data to mitigate their effects. Similarly, Hidden Confounder Removal (HCR) [43] employs front-door adjustment to address latent confounders, while iDCF [37] utilises proximal causal inference techniques for the same purpose. However, these methods may struggle to infer latent confounders directly from interaction data, as some latent confounders may lack proxy variables in data.

Another line of research, exemplified by IV4REC [28], leverages the pre-defined IVs [4, 12, 30] to decompose the input vectors of recommendation models and mitigate the effects of latent confounders between items and user feedback. However, these approaches directly modify the input vectors while overlooking the latent confounders between item exposure and user feedback, leaving more complex types of latent confounding unresolved.

In contrast, our IViDR method address both types of latent confounders in recommender systems by developing a novel IV-based approach combined with an iVAE. Note that our IViDR method effectively handles dual latent confounding biases, with or without proxy variables, resulting in a reliable and unbiased recommender system, as demonstrated in our experiments.

---

## 3 Preliminaries

In this section, we introduce important notations, definitions, and the concept of IV that are used throughout the paper.

### Table 1: Definitions and Notations

| Symbol | Definition |
| :--- | :--- |
| $\mathcal{U}$ | Users. |
| $\mathcal{I}$ | Items. |
| $\mathbf{Z}$ | The instrumental variable (the embeddings of user features). |
| $\mathbf{T}$ | The treatment (the embedding of the target item $i$ and the embeddings of the items interacted with $u$). |
| $\mathbf{A}$ | Exposure vector. |
| $\mathbf{R}$ | User feedback. |
| $\mathbf{W}$ | The set of proxy variables. |
| $\mathbf{C}$ | Latent confounders with proxy variables. |
| $\mathbf{B}$ | Latent confounders without proxy variables. |
| $\mathbf{X}$ | The user-item interaction data. |

### 3.1 Notations

The primary notations are summarised in Table 1. We represent vectors with bold-faced uppercase (e.g., $\mathbf{X}$), and their elements by lowercase letters with subscripts (e.g., $x_{ij}$).

Let $\mathcal{U}$ represent the set of users, where $|\mathcal{U}| = m$ denotes the total number of users. Similarly, let $\mathcal{I}$ represent the set of items, with $|\mathcal{I}| = n$ denoting the total number of items. For each user $u \in \mathcal{U}$, the exposure vector $\mathbf{A}$ is defined as $\mathbf{A} = [a_{u1}, a_{u2}, \dots, a_{un}] \in \{0, 1\}^n$, where $a_{ui} = 1$ indicates that item $i$ is exposed to user $u$, and $a_{ui} = 0$ indicates it is not. The feedback from user $u$ is represented by the vector $\mathbf{R} = [r_{u1}, r_{u2}, \dots, r_{un}]$, where $r_{ui}$ denotes the observed feedback from user $u$ for item $i$.

In this work, we adopt the potential outcomes framework [14] to develop debiased recommender systems. Let $r_{ui}^a$ denote the potential outcome for user $u$ on item $i$ under the exposure status $a$. Previous studies [32] have assumed that the exposure of item $i$ to user $u$ is the sole factor influencing this outcome. However, our method accounts for additional covariates or latent variables that may affect the outcomes $r_{ui}^a$. The set of observed proxy variables for user $u$ is represented by $\mathbf{W}$. The set of user features is indicated by $\mathbf{Z}$. The recommendation dataset $\mathcal{D}$ consists of the interaction data $\mathbf{X}$, the set of user features $\mathbf{Z}$, and the set of proxy variables $\mathbf{W}$. The probability that user $u$ will provide positive feedback on item $i$, given the exposure vector $\mathbf{A}$, is denoted by $p(r_{ui} = 1 \mid \mathbf{A})$. In recommender systems, the set of treatment variables $\mathbf{T}$ is defined as a set of embeddings, which includes the embedding of the target item $i$ and the embeddings of the items previously interacted with by user $u$, to predict the preference score.

Moreover, we use a graphical causal model [5, 24], specifically a directed acyclic graph (DAG), to represent the causal relationships between variables in recommender systems. Our proposed causal DAG $\mathcal{G}$ is illustrated in Figure 1. In the DAG, $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, $\mathcal{V} = \mathbf{Z} \cup \mathbf{W} \cup \mathbf{C} \cup \mathbf{B} \cup \{\mathbf{T}, \mathbf{A}, \mathbf{R}\}$ is the set of vertices and $\mathcal{E}$ represents the set of directed edges. Here, $\mathbf{B}$ denotes latent confounders without proxy variables, while $\mathbf{C}$ represents latent confounders with corresponding proxy variables.

```
          [ Z ] (Observed IV)
            |
            v
   ( B )-->[ T ] (Treatment) ------------> [ R ] (Outcome)
     |                                      ^   ^
     +--------------------------------------+   |
                                                |
   [ W ] <-- ( C ) (Latent Confounder) ---------+
               |
               v
             [ A ] (Exposure Status)
```
*Figure 1: An example DAG illustrating that $\mathbf{Z}$ serves as an IV. $\mathbf{T}$ and $\mathbf{R}$ represent the treatment and outcome variables, respectively, while $\mathbf{B}$ denotes latent confounders without proxy variables. $\mathbf{C}$ denotes latent confounders with proxy variables, $\mathbf{A}$ indicates exposure status, and $\mathbf{W}$ represents the set of proxy variables.*

### 3.2 Instrumental Variable (IV) Approach

The IV approach [4, 12, 30] is a useful tool for addressing the confounding bias caused by latent confounders. The set of variable $\mathbf{Z}$ is a set of valid IVs relative to pair $\mathbf{T}$ and $\mathbf{R}$ when it satisfies the following three conditions:

> **Definition 1 (Instrumental Variable (IV) [24]).** *Given a DAG $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where the set of vertices is defined as $\mathcal{V} = \mathbf{Z} \cup \mathbf{W} \cup \mathbf{C} \cup \mathbf{B} \cup \{\mathbf{T}, \mathbf{A}, \mathbf{R}\}$ and the set of directed edges as $\mathcal{E}$, then $\mathbf{Z}$ is a set of valid IVs if the following conditions hold:*
> 1. $\mathbf{Z} \not\!\perp\!\!\!\perp_d \mathbf{T}$, *meaning there is at least one unblocked path between $\mathbf{Z}$ and $\mathbf{T}$ in $\mathcal{G}$.*
> 2. $\mathbf{Z} \perp\!\!\!\perp_d \mathbf{R} \mid \{\mathbf{T}, \mathbf{B}\}$ *in $\mathcal{G}_{\overline{\mathbf{T}}}$ (obtained by removing the edge $\mathbf{T} \to \mathbf{R}$ from $\mathcal{G}$).*
> 3. $\mathbf{Z}$ *does not share confounders with $\mathbf{R}$, i.e., $(\mathbf{Z} \perp\!\!\!\perp_d \mathbf{R})_{\mathcal{G}_{\underline{\mathbf{T}}}}$.*

where $\not\!\perp\!\!\!\perp_d$ and $\perp\!\!\!\perp_d$ refer to d-connection and d-separation, respectively [24]. Once a variable satisfies the three conditions, it can serve as a valid IV. Existing IV-based methods can be employed to mitigate the confounding bias caused by latent confounders in causal effect estimation. Most IV-based methods utilize the two-stage least squares (2SLS) procedure [17]. Recently, several IV-based causal learning approaches have extended the 2SLS by incorporating deep learning techniques. For example, DeepIV [12] provides a flexible framework that combines deep learning with the 2SLS method. DFIV [35] introduces an alternating training strategy for 2SLS, demonstrating strong performance on high-dimensional data.

Our problem setting is illustrated in the causal DAG shown in Figure 1, where the set of latent confounders $\mathbf{B}$ exist between $\mathbf{T}$ and $\mathbf{R}$. In this setting, $\mathbf{Z}$ (the embeddings of user features, such as observed gender) serves as a valid IV. Consequently, we employ IV-based methods [3, 8] to estimate the causal effect of $\mathbf{T}$ on $\mathbf{R}$. Specifically, we use the 2SLS method to reconstruct $\mathbf{T}$ and mitigate the effects of latent confounders $\mathbf{B}$.

---

## 4 The Proposed IViDR Method

In this section, we provide the details of our proposed IViDR method. First, we present an overview of the IViDR method. Next, we provide a theoretical analysis of the correctness of IVs in the IViDR method. We then introduce the 2SLS method in IViDR to reconstruct treatments and obtain debiased interaction data for $\mathbf{T}$ and $\mathbf{R}$. After that, we describe how latent confounders between item exposure $\mathbf{A}$ and $\mathbf{R}$ are learned using an iVAE within the IViDR method. Additionally, we prove the identifiability of the learned representations. Finally, we present the objective function of the IViDR method.

### 4.1 An Overview for IViDR Method

The main steps of the IViDR method are illustrated in Figure 2. In IViDR, we use the embeddings of user features, $\mathbf{Z}$, as instrumental variables (IVs). The details of the IViDR are as follows: First, IViDR uses IVs to decompose the treatments into fitted (which explain why a user prefers an item) and residual components (which capture other factors influencing the user-item interaction). With the aid of IVs, we reconstruct $\mathbf{T}$ to obtain the reconstruct treatment $\mathbf{T}^{\text{re}}$, ensuring that $\mathbf{T}^{\text{re}}$ is not affected by the latent confounders $\mathbf{B}$. The reconstruct treatment $\mathbf{T}^{\text{re}}$ is then combined with the interaction data to generate the debiased interaction data. Next, IViDR employs the iVAE to efficiently learn identifiable latent representations from a combination of proxy variables, interaction data, and debiased interaction data, mitigating the effects of latent confounders $\mathbf{C}$. Finally, IViDR adjusts for the learned representations to further reduce confounding biases. Additionally, we provide a theoretical analysis of the identifiability of learning the representation $\mathbf{C}$ through our IViDR method. Due to space limitations, the pseudo-code of IViDR is provided in Appendix B.

*Figure 2: The architecture of our proposed IViDR method. IViDR uses the embeddings of user features as IVs to decompose treatments into fitted (the values fitted by the regression model) and residual (the residuals) components. It then reconstructs the treatments and combines them with interactional data to generate debiased interactional data. An iVAE is employed to infer latent representations from proxy variables, interactional data, and debiased interactional data. Finally, IViDR adjusts for these latent representations to mitigate confounding biases.*

### 4.2 The Soundness of IV in IViDR Method

In recommender systems, confounding bias typically arises from specific interactions when users access the service. However, since the embeddings of user features represent all users, it is reasonable to conclude that there is no confounding bias between $\mathbf{Z}$ and $\mathbf{R}$. Thus, we utilise the embeddings of user features as a valid IV. Furthermore, we provide a theoretical analysis to demonstrate the soundness of using the embeddings of user features, $\mathbf{Z}$, as a valid IV for mitigating the confounding bias caused by latent confounders:

> **Theorem 1.** *Given a causal DAG $\mathcal{G} = (\mathbf{Z} \cup \mathbf{W} \cup \mathbf{C} \cup \mathbf{B} \cup \{\mathbf{T}, \mathbf{A}, \mathbf{R}\}, \mathcal{E})$, where $\mathbf{T}$ and $\mathbf{R}$ are the treatment and outcome, respectively, and $\mathcal{E}$ is the set of edges between the variables. Let $\mathbf{B}$ denote the latent confounders between $\mathbf{T}$ and $\mathbf{R}$. Suppose there exists a directed edge $\mathbf{T} \to \mathbf{R}$ in $\mathcal{E}$, and $\mathbf{Z}$ represents the embeddings of user features. Additionally, assume that the underlying data generating process is faithful to the proposed causal DAG $\mathcal{G}$ as shown in Figure 1. Therefore, $\mathbf{Z}$ serves as the valid IV for estimating the causal effect of $\mathbf{T}$ on $\mathbf{R}$.*

The proof can be found in Appendix A. Theorem 1 allows us to use the embeddings of user features, $\mathbf{Z}$, as valid IVs to decompose the treatment variable, $\mathbf{T}$, for obtaining the reconstructed treatment.

### 4.3 Interaction Data Reconstruction

We introduce the treatment reconstruction component using the IV (i.e., the embeddings of user features $\mathbf{Z}$) in our IViDR method.

#### 4.3.1 Construction of treatments and IVs

The treatment variable $\mathbf{T}$ in our problem setting is defined as:
$$\mathbf{T} = \left\{ \mathbf{T}_j : j \in \mathcal{I}_u \cup \{i\} \right\} \tag{1}$$
where $\mathbf{T}_j \in \mathbb{R}^{d_i}$ is the embedding vector of item $j$, and $\mathcal{I}_u$ represents the set of items interacted with by user $u$ in the recommendation dataset $\mathcal{D}$. $\mathcal{I}_u$ is defined as:
$$\mathcal{I}_u = \left\{ i' : \exists (u, i', r_{ui'}) \in \mathcal{D} \right\} \tag{2}$$

We define a set of matrices $\mathbf{Z}_j$, where each matrix $\mathbf{Z}_j$ represents the embeddings of user features related to item $j$ as follows:
$$\mathbf{Z} = \left\{ \mathbf{Z}_j : j \in \mathcal{I}_u \cup \{i\} \right\} \tag{3}$$

Hence, we have the treatment variable $\mathbf{T}$ and the embeddings of user features, $\mathbf{Z}$, for building the debiased recommendation system.

#### 4.3.2 Treatment reconstruction

At this stage, we reconstruct the treatment $\mathbf{T}$ using the IV $\mathbf{Z}$ to obtain the reconstructed treatment $\mathbf{T}^{\text{re}}$, which mitigates the confounding bias caused by latent confounders $\mathbf{B}$ by applying the idea in 2SLS method. This process can be broken down into two steps: first, treatment decomposition, and second, treatment combination.

**Step I: Treatment decomposition.** We regress $\mathbf{T}$ to get the fitted part of the regression $\widehat{\mathbf{T}}$:
$$\widehat{\mathbf{T}} = \left\{ \widehat{\mathbf{T}}_j = f_{\text{proj}}(\mathbf{T}_j, \mathbf{Z}_j) : j \in \mathcal{I}_u \cup \{i\} \right\} \tag{4}$$
where $\mathbf{T}_j \in \mathbf{T}$ and $\mathbf{Z}_j \in \mathbf{Z}$, and $f_{\text{proj}} : \mathbb{R}^{d_i} \times \mathbb{R}^{d_q \times N} \to \mathbb{R}^{d_q}$ is defined as the product of the matrix $\mathbf{Z}_j$ with an $N$-dimensional vector $\boldsymbol{\tau}_j$:
$$f_{\text{proj}}(\mathbf{T}_j, \mathbf{Z}_j) = \mathbf{Z}_j \boldsymbol{\tau}_j \tag{5}$$
where $\boldsymbol{\tau}_j$ is the closed-form solution of a least squares regression. $\boldsymbol{\tau}_j$ can be obtained by the following equation:
$$\boldsymbol{\tau}_j = \arg\min_{\boldsymbol{\tau}_j \in \mathbb{R}^N} \|\mathbf{Z}_j \boldsymbol{\tau}_j - \text{MLP}_0(\mathbf{T}_j)\|_2^2 = \mathbf{Z}_j^\dagger \text{MLP}_0(\mathbf{T}_j) \tag{6}$$
where $\mathbf{Z}_j^\dagger$ is the Moore-Penrose pseudoinverse of $\mathbf{Z}_j$, and $\text{MLP}_0 : \mathbb{R}^{d_i} \to \mathbb{R}^{d_q}$ is a multi-layer perceptron. We refer to $\widehat{\mathbf{T}}_j \in \widehat{\mathbf{T}}$ as the fitted part of the embedding $\mathbf{T}_j$. Then, we obtain the residual part of the regression $\widetilde{\mathbf{T}}$ as:
$$\widetilde{\mathbf{T}} = \left\{ \widetilde{\mathbf{T}}_j = \text{MLP}_0(\mathbf{T}_j) - \widehat{\mathbf{T}}_j : j \in \mathcal{I}_u \cup \{i\} \right\} \tag{7}$$

**Step II: Treatment combination.** We obtain the reconstructed treatment $\mathbf{T}^{\text{re}}$ by combining $\widehat{\mathbf{T}}$ and $\widetilde{\mathbf{T}}$ as follows:
$$\mathbf{T}^{\text{re}} = \left\{ \mathbf{T}_j^{\text{re}} = \alpha_{1j} \widehat{\mathbf{T}}_j + \alpha_{2j} \widetilde{\mathbf{T}}_j : j \in \mathcal{I}_u \cup \{i\} \right\} \tag{8}$$
where $\widehat{\mathbf{T}}_j \in \widehat{\mathbf{T}}$ and $\widetilde{\mathbf{T}}_j \in \widetilde{\mathbf{T}}$, and $\alpha_{1j} \in \mathbb{R}$ and $\alpha_{2j} \in \mathbb{R}$ are two weights, which are estimated by two multi-layer perceptrons (MLPs):
$$\alpha_{1j} = \text{MLP}_1([\text{MLP}_0(\mathbf{T}_j), \mathbf{Z}_j]), \quad \alpha_{2j} = \text{MLP}_2([\text{MLP}_0(\mathbf{T}_j), \mathbf{Z}_j]) \tag{9}$$
where the inputs to the two MLPs are concatenations of the transformed $\mathbf{T}_j$ and $\mathbf{Z}_j$, corresponding to item $j$.

#### 4.3.3 Interaction data reconstruction

Finally, we reconstruct the interaction data $\mathbf{X}$ using $\mathbf{T}^{\text{re}}$ to obtain the debiased interaction data $\mathbf{X}^{\text{re}}$, which serves as the input for iVAE in our IViDR method:
$$\mathbf{X}^{\text{re}} = \mathbf{X} + \mathbf{T}_j^{\text{re}} \tag{10}$$

### 4.4 Learning the Latent Representation Using iVAE in IViDR Method

In the previous section, we employed the IV approach to address the confounding bias caused by latent confounders $\mathbf{B}$, but the latent confounders $\mathbf{C}$ still bias the recommender system. To mitigate this, we propose using a proxy variable $\mathbf{W}$ (e.g., the mean price of consumed goods) to recover the representation of latent confounders $\mathbf{C}$.

We assume that $\mathbf{W}$ is a Bernoulli random variable with mean $\mu(\mathbf{C}) \in (0, 1)$ and is correlated with $\mathbf{C}$ given the exposure vector $\mathbf{A}$. The probability $p(r_{ui} = 1 \mid \mathbf{A}, \mathbf{W})$, representing the likelihood that user $u$ gives positive feedback on item $i$ given $\mathbf{A}$ and $\mathbf{W}$, can be inferred from the available dataset. We further assume that $p(\mathbf{C} \mid \mathbf{A}, \mathbf{W})$, the probability distribution of the estimated latent representation $\mathbf{C}$ given $\mathbf{A}$ and $\mathbf{W}$, is uniquely defined by the factor model [19]. Next, we show how to identify $p(r_{ui}^a)$ with proxy variables in general, noting that:
$$p(r_{ui}^a) = \mathbb{E}_{\mathbf{C}}[p(r_{ui} \mid \mathbf{A}, \mathbf{C})] = \int_c p(\mathbf{C} = c) p(r_{ui} \mid \mathbf{A}, \mathbf{C} = c) \, dc \tag{11}$$

From Eq. (11), it is clear that the key to estimating $p(r_{ui}^a)$ lies in determining both $p(\mathbf{C})$ and $p(r_{ui} \mid \mathbf{A}, \mathbf{C})$. This process can be broken down into two steps: first, estimating $p(\mathbf{C})$, and second, determining $p(r_{ui} \mid \mathbf{A}, \mathbf{C})$.

**Step I:** The first step is to obtain $p(\mathbf{C})$. This involves learning $\mathbf{C}$ from $\mathbf{W}$, ensuring that the learned $\mathbf{C}$ closely approximates the true latent confounders $\mathbf{C}$ [15, 23] as follows:
$$p(\mathbf{C} = c) = \mathbb{E}_{\mathbf{A}, \mathbf{W}}[p(\mathbf{C} \mid \mathbf{A}, \mathbf{W})] \tag{12}$$
Hence, we need to learn $p(\mathbf{C} \mid \mathbf{A}, \mathbf{W})$ from the data. In practice, we use the following equation:
$$p(\mathbf{A} \mid \mathbf{W}) = \int_c p(\mathbf{A} \mid \mathbf{C} = c) p(\mathbf{C} = c \mid \mathbf{A}, \mathbf{W}) \, dc \tag{13}$$
where $p(\mathbf{A} \mid \mathbf{W})$ can be learned directly from the data. Therefore, we need to recover $p(\mathbf{C} \mid \mathbf{A}, \mathbf{W})$ and $p(\mathbf{A} \mid \mathbf{C})$ from the data. In our IViDR framework, we use iVAE [15] to recover both terms.

**Step II:** In the second step, we obtain $p(r_{ui} \mid \mathbf{A}, \mathbf{C})$, which can be inferred from $p(\mathbf{C} \mid \mathbf{A}, \mathbf{W})$ and $p(r_{ui} \mid \mathbf{C} = c, \mathbf{A})$ as follows:
$$p(r_{ui} \mid \mathbf{A}, \mathbf{W}) = \int_c p(r_{ui} \mid \mathbf{A}, \mathbf{C} = c) p(\mathbf{C} = c \mid \mathbf{A}, \mathbf{W}) \, dc \tag{14}$$

In IViDR, we also use iVAE to recover $p(r_{ui} \mid \mathbf{C} = c, \mathbf{A})$. With these two steps, we can obtain $p(\mathbf{C})$ and $p(r_{ui} \mid \mathbf{A}, \mathbf{C})$. By applying Eq. (11), we can determine $p(r_{ui}^a)$, the probability distribution of the potential outcome, from the data.

We use the debiased interactional data $\mathbf{X}^{\text{re}}$ as input to iVAE. To learn $p_\theta(\mathbf{C}_1 \mid \mathbf{A}, \mathbf{W})$ (the details of obtaining the approximate posterior of latent confounders can be found in Appendix D.1), we use $q_\phi(\mathbf{C}_1 \mid \mathbf{A}, \mathbf{W})$ as the approximate posterior. We sample the latent confounder $\mathbf{C}_1$ from $q_\phi(\mathbf{C}_1 \mid \mathbf{A}, \mathbf{W})$. Next, we use the interactional data $\mathbf{X}$ as input to iVAE. To learn $p_\theta(\mathbf{C}_2 \mid \mathbf{A}, \mathbf{W})$, we use $q_\phi(\mathbf{C}_2 \mid \mathbf{A}, \mathbf{W})$ as the approximate posterior. We sample the latent confounder $\mathbf{C}_2$ from $q_\phi(\mathbf{C}_2 \mid \mathbf{A}, \mathbf{W})$. Finally, we fuse $\mathbf{C}_1$ and $\mathbf{C}_2$ to obtain latent confounder $\mathbf{C}$:
$$\mathbf{C} = \rho * \mathbf{C}_1 + \tau * \mathbf{C}_2 \tag{15}$$
where $\rho$ and $\tau$ are tuning parameters.

Thus, we can recover the latent confounder $\mathbf{C}$, and subsequently implement our iVAE submodel within the IViDR method:
$$\mathcal{L}_{iVAE} = \mathbb{E}_{\rho * q_\phi(\mathbf{C}_1 \mid \mathbf{A}, \mathbf{W}) + \tau * q_\phi(\mathbf{C}_2 \mid \mathbf{A}, \mathbf{W})} \tag{16}$$

### 4.5 The Identifiability of the Learned Representation

In this section, we prove that the estimated latent representation $\mathbf{C}$ learned from $\mathbf{X}^{\text{re}}$ is identifiable. The identifiability of $\mathbf{C}$ learned by our IViDR is proved as follows.

Let $\theta = (\mathbf{f}, \mathbf{H}, \boldsymbol{\lambda})$ be the parameter of the model $\Theta$ that generates the following condition:
$$p_\theta(\mathbf{X}^{\text{re}}, \mathbf{C} \mid \mathbf{W}) = p_{\mathbf{f}}(\mathbf{X}^{\text{re}} \mid \mathbf{C}) p_{\mathbf{H}, \boldsymbol{\lambda}}(\mathbf{C} \mid \mathbf{W}) \tag{17}$$
where $p_{\mathbf{f}}(\mathbf{X}^{\text{re}} \mid \mathbf{C})$ is defined as:
$$p_{\mathbf{f}}(\mathbf{X}^{\text{re}} \mid \mathbf{C}) = p_{\boldsymbol{\varepsilon}}(\mathbf{X}^{\text{re}} - \mathbf{f}(\mathbf{C})) \tag{18}$$
where the value of $\mathbf{X}^{\text{re}}$ is decomposed as $\mathbf{X}^{\text{re}} = \mathbf{f}(\mathbf{C}) + \boldsymbol{\varepsilon}$. Here, $\boldsymbol{\varepsilon}$ is an independent noise variable (independent of both $\mathbf{C}$ and $\mathbf{f}$), and its probability density function is given by $p_{\boldsymbol{\varepsilon}}(\boldsymbol{\varepsilon})$.

As stated in Assumption 1, we assume that the conditional distribution $p_{\mathbf{H}, \boldsymbol{\lambda}}(\mathbf{C} \mid \mathbf{W})$ is a conditional factor distribution and follows an exponential family of distributions, illustrating the relationship between the latent representation $\mathbf{C}$ and the proxy variables $\mathbf{W}$.

> **Assumption 1.** *The conditioning on $\mathbf{W}$ utilizes an arbitrary function—like a look-up table or a neural network—that produces the specific exponential family parameters $\boldsymbol{\lambda}_{i,j}$. Consequently, the probability density function is expressed as follows:*
> $$p_{\mathbf{H}, \boldsymbol{\lambda}}(\mathbf{C} \mid \mathbf{W}) = \prod_i \frac{Q_i(C_i)}{Z_i(\mathbf{W})} \exp \left[ \sum_{j=1}^k H_{i,j}(C_i) \lambda_{i,j}(\mathbf{W}) \right] \tag{19}$$
> *where $Q_i$ is the base measure, $Z_i(\mathbf{W})$ is the normalization constant, $\boldsymbol{\lambda}_i(\mathbf{W}) = (\lambda_{i,1}(\mathbf{W}), \dots, \lambda_{i,k}(\mathbf{W}))$ is a parameter related to $\mathbf{W}$, $\mathbf{H}_i = (H_{i,1}, \dots, H_{i,k})$ is the sufficient statistic, and $k$ is the dimension of each sufficient statistic, held constant.*

To prove the identifiability of the learned representation $\mathbf{C}$ in our IViDR, we introduce the following definitions [15]:

> **Definition 2 (Identifiability classes).** *Let $\sim$ be an equivalence relation on $\Theta$. We say that Eq. (17) is identifiable (or $\sim$ identifiable) under $\sim$ if:*
> $$p_\theta(\mathbf{X}^{\text{re}}) = p_{\widetilde{\theta}}(\mathbf{X}^{\text{re}}) \implies \widetilde{\theta} \sim \theta \tag{20}$$
> *$\Theta / \sim$ are called identifiability classes.*

We now define the equivalence relation on the parameter set $\Theta$:

> **Definition 3.** *Let $\sim_M$ be the equivalence relation defined on $\Theta$ as follows:*
> $$(\mathbf{f}, \mathbf{H}, \boldsymbol{\lambda}) \sim (\widetilde{\mathbf{f}}, \widetilde{\mathbf{H}}, \widetilde{\boldsymbol{\lambda}}) \iff \exists M, \mathbf{q} \mid \mathbf{H}(\mathbf{f}^{-1}(\mathbf{X}^{\text{re}})) = M \widetilde{\mathbf{H}}(\widetilde{\mathbf{f}}^{-1}(\mathbf{X}^{\text{re}})) + \mathbf{q}, \quad \forall \mathbf{X}^{\text{re}} \in \mathcal{X} \tag{21}$$
> *where $M$ is an $nk \times nk$ matrix and $\mathbf{q}$ is a vector.*

The identifiability of the learned representation $\mathbf{C}$ in our IViDR can be derived as follows:

> **Theorem 2.** *Assuming we have data collected from a generative model as defined by Eqs. (17)–(19), with parameters $(\mathbf{f}, \mathbf{H}, \boldsymbol{\lambda})$. Suppose the following conditions hold:*
> 1. *The set $\{\mathbf{X}^{\text{re}} \in \mathcal{X} \mid \varphi_{\boldsymbol{\varepsilon}}(\mathbf{X}^{\text{re}}) = 0\}$ has measure zero, where $\varphi_{\boldsymbol{\varepsilon}}$ is the characteristic function of the density $p_{\boldsymbol{\varepsilon}}$ defined in Eq. (18).*
> 2. *The hybrid function $\mathbf{f}$ in Eq. (18) is injective.*
> 3. *The sufficient statistic $H_{i,j}$ in Eq. (19) is differentiable almost everywhere, and $(H_{i,j})_{1 \le j \le k}$ is linearly uncorrelated on any subset of $\mathcal{X}$ that has a measure greater than zero.*
> 4. *There exist $nk + 1$ distinct points $\mathbf{W}_0, \dots, \mathbf{W}_{nk}$ such that the matrix*
> $$L = (\boldsymbol{\lambda}(\mathbf{W}_1) - \boldsymbol{\lambda}(\mathbf{W}_0), \dots, \boldsymbol{\lambda}(\mathbf{W}_{nk}) - \boldsymbol{\lambda}(\mathbf{W}_0)) \tag{22}$$
> *of size $nk \times nk$ is invertible.*
> 
> *Then the parameters $(\mathbf{f}, \mathbf{H}, \boldsymbol{\lambda})$ are $\sim_M$ identifiable.*

The proof is structured into three main steps:
- In the first step, we apply the simple convolution technique, as permitted by Assumption (i), to transform the equation for the debiased interaction data distribution into one representing the noise-free distribution. This effectively reduces the noisy scenario to a noise-free case, leading to Eq. (34).
- In the second step, we eliminate all terms related to the debiased interaction data $\mathbf{X}^{\text{re}}$ and the proxy variable $\mathbf{W}$. This is accomplished by leveraging the conditions provided by assumption (iv) and using $\mathbf{W}_0$ as the pivot. This is detailed in Eqs. (34)–(38).
- Finally, we demonstrate that the linear transformation is invertible, establishing an equivalence relation. This final step relies on Assumption (iii).

Detailed proofs of the three steps can be found in Appendix C.

### 4.6 The Proposed IViDR Framework

We propose a generalized framework that is compatible with any recommendation model. For comparison, we select the Matrix Factorisation (MF) model as the recommendation model for our framework. The predicted rating $\mathcal{L}_{MF}$ is given by:
$$\mathcal{L}_{MF} = \mathbf{p}_u^T \mathbf{q}_i + k_u + k_i \tag{23}$$
where $\mathbf{p}_u$ is the latent vector for user $u$, $\mathbf{q}_i$ is the latent vector for item $i$, and $T$ denotes the transpose operation. Furthermore, $k_u$ represents the user preference bias term, and $k_i$ is the item preference bias.

Finally, we utilise a point-wise recommendation model parameterised by $\eta$ to estimate $p(r_{ui} \mid \mathbf{A}, \mathbf{C})$. In particular, we utilize an additive model $f(u, i, \mathbf{C}; \eta)$. The additive model is:
$$f(u, i, \mathbf{C}; \eta) = \phi * \mathcal{L}_{iVAE} + \lambda * \mathcal{L}_{MF} \tag{24}$$
where $\phi, \lambda$ are tuning parameters. Therefore, our final loss function for IViDR is defined as:
$$\mathcal{L}_{IViDR} = \frac{1}{|\mathcal{D}|} \sum_{(u, i) \in \mathcal{D}} l\left( \mathbb{E}_{\rho * q_\phi(\mathbf{C}_1 \mid \mathbf{A}, \mathbf{W}) + \tau * q_\phi(\mathbf{C}_2 \mid \mathbf{A}, \mathbf{W})} [f(u, i, \mathbf{C}; \eta)], \, r_{ui} \right) \tag{25}$$
where $l(\cdot, \cdot)$ denotes the binary cross-entropy (BCE) loss, which is commonly used in such models.

**Limitations.** The success of IViDR relies on the assumptions of IVs and the existence of proxy variables in the data. However, the relationship between user preferences, features, and recommendation mechanisms is often complex, making it challenging to validate the assumptions and ensure that appropriate IVs and proxy variables are available. Additionally, the success of IViDR depends on the quality and relevance of these variables, which may not always be guaranteed in real-world datasets.

---

## 5 Experiments

In this section, we perform extensive experiments to evaluate the effectiveness of our IViDR method within recommender systems, especially when latent confounders are present.

### 5.1 Experimental Settings

We begin by introducing the datasets, followed by descriptions of the baseline methods and evaluation metrics. Lastly, we present the parameter settings used for the IViDR model.

**Dataset description.** We conduct experiments on three real-world datasets, summarized in Table 2. Details of the datasets can be found in Appendix E.1.

### Table 2: The statistics of Coat, Yahoo!R3, and KuaiRand

| Dataset | #User | #Item | #Biased Data | #Unbiased Data |
| :--- | :---: | :---: | :---: | :---: |
| Coat | 290 | 300 | 6,960 | 4,640 |
| Yahoo! R3 | 5,400 | 1,000 | 129,179 | 54,000 |
| KuaiRand | 23,533 | 6,712 | 1,413,574 | 954,814 |

**Baselines.** We compare our method with several state-of-the-art (SOTA) deconfounding methods:
1. **MF [18] & MF with features (MF-WF):** MF decomposes a rating matrix into the product of two lower dimensional matrices. MF-WF is an improved version of MF.
2. **IPS [27] & RD-IPS [9]:** IPS estimates causal effects in observations, and reduces selection bias by IPW. RD-IPS is an improved version of IPS.
3. **InvPref [33]:** InvPref decouples a user's true preferences from biased behaviour and mitigates bias using invariant learning.
4. **DDCF-MF [44]:** DeepDCF combines deep learning and collaborative filtering techniques to predict user preferences for items. DDCF-MF uses MF as the backbone.
5. **IV4R-MF [28]:** IV4Rec applies IVs to decompose input vectors within recommendation models to mitigate the effects of latent confounders. IV4R-MF uses MF as the backbone.
6. **iDCF [37]:** The iDCF incorporates proximal causal inference techniques to identifiably predict users' counterfactual feedback on items.

**Evaluation metrics.** We evaluate the top five recommendations using RECALL@5 and NDCG@5 metrics. Each method is tested across ten trials, and we report the average and standard deviation to demonstrate the effectiveness of each method.

**Implementation and Configuration.** We implement our IViDR model in PyTorch. A grid search is performed to determine the optimal hyperparameters for each method. The learning rate is selected from the set $\{1\text{e-}3, 5\text{e-}4, 1\text{e-}4, 5\text{e-}5, 1\text{e-}5\}$, and the weight decay is chosen from $\{1\text{e-}5, 1\text{e-}6\}$. To ensure a fair comparison, we utilize the Adam optimizer [16] for all methods, including our IViDR method.

We set the parameters $\phi$ and $\lambda$ in Eq. (24) to $1$, and $\rho$ and $\tau$ in Eq. (15) to $0.9$. The treatment embedding and user feature embeddings are $32$ dimensions in the Coat dataset, $96$ dimensions in the Yahoo!R3 dataset, and $128$ dimensions in the KuaiRand dataset.

### 5.2 Performance Comparison

Table 3 presents the experimental results for each method, which are also visualized in Figure 4 (with Figure 4 located in Appendix G.1). The results demonstrate that IViDR significantly outperforms all other algorithms across all datasets and evaluation metrics. 

### Table 3: Performance metrics across three real-world datasets

*The best method is highlighted in bold, and the second-best is underlined. We report the p-values from t-tests to assess the statistical significance of the performance differences.*

| Datasets | Coat NDCG@5 | Coat RECALL@5 | Yahoo!R3 NDCG@5 | Yahoo!R3 RECALL@5 | KuaiRand NDCG@5 | KuaiRand RECALL@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MF | $0.5590 \pm 0.0126$ | $0.5418 \pm 0.0116$ | $0.5624 \pm 0.0087$ | $0.7125 \pm 0.0076$ | $0.3732 \pm 0.0011$ | $0.3244 \pm 0.0008$ |
| MF-WF | $0.5577 \pm 0.0120$ | $0.5336 \pm 0.0102$ | $0.5617 \pm 0.0087$ | $0.7120 \pm 0.0127$ | $0.3710 \pm 0.0006$ | $0.3238 \pm 0.0009$ |
| IPS | $0.5511 \pm 0.0154$ | $0.5358 \pm 0.0113$ | $0.5500 \pm 0.0037$ | $0.6961 \pm 0.0063$ | $0.3698 \pm 0.0014$ | $0.3234 \pm 0.0011$ |
| RD-IPS | $0.5484 \pm 0.0144$ | $0.5302 \pm 0.0176$ | $0.5334 \pm 0.0116$ | $0.6796 \pm 0.0146$ | $0.3469 \pm 0.0009$ | $0.3116 \pm 0.0012$ |
| InvPref | $0.5372 \pm 0.0110$ | $0.5247 \pm 0.0097$ | $0.5881 \pm 0.0037$ | $0.7388 \pm 0.0044$ | $0.3797 \pm 0.0010$ | $0.3282 \pm 0.0005$ |
| DDCF-MF | $0.5631 \pm 0.0068$ | $0.5425 \pm 0.0107$ | $0.6260 \pm 0.0064$ | $0.7636 \pm 0.0052$ | $0.3959 \pm 0.0007$ | $0.3456 \pm 0.0009$ |
| IV4R-MF | $0.5617 \pm 0.0099$ | $0.5433 \pm 0.0097$ | $0.5653 \pm 0.0062$ | $0.7152 \pm 0.0085$ | $0.3741 \pm 0.0010$ | $0.3251 \pm 0.0008$ |
| iDCF | $\underline{0.5638 \pm 0.0124}$ | $\underline{0.5493 \pm 0.0111}$ | $\underline{0.6410 \pm 0.0022}$ | $\underline{0.7780 \pm 0.0039}$ | $\underline{0.4080 \pm 0.0004}$ | $\underline{0.3481 \pm 0.0008}$ |
| **IViDR** | $\mathbf{0.5903 \pm 0.0101}$ | $\mathbf{0.5783 \pm 0.0114}$ | $\mathbf{0.6602 \pm 0.0038}$ | $\mathbf{0.7901 \pm 0.0037}$ | $\mathbf{0.4161 \pm 0.0004}$ | $\mathbf{0.3549 \pm 0.0006}$ |
| **p-value** | $6\text{e-}5$ | $1\text{e-}5$ | $1\text{e-}9$ | $1\text{e-}6$ | $5\text{e-}20$ | $1\text{e-}13$ |

From Table 3 and Figure 4, we observe the following:
1. Our proposed IViDR achieves the best performance across all datasets and metrics. The p-values indicate that the performance improvement of IViDR over other methods is statistically significant. This demonstrates that IViDR effectively enhances recommendation accuracy and reduces bias.
2. iDCF across all datasets indicates its effectiveness in mitigating confounding bias through the use of latent confounders.
3. IV4R-MF exhibits clear improvements over the basic MF method. By leveraging instrumental variables, IV4R-MF enhances the accuracy of recommendations across multiple datasets.
4. DDCF-MF shows notable improvements over the basic MF and MF-WF methods, especially on the Yahoo!R3 and KuaiRand datasets. The incorporation of deep learning components allows DDCF-MF to effectively capture complex feature interactions, enhancing recommendation accuracy and robustness.

### 5.3 Ablation Study

To further understand the impact of each component in our model, we performed ablation studies. We verified the following cases:
1. The case where the original treatment $\mathbf{T}$ is added to the interactional data (**IViDR-T**).
2. The case where only the fitted part $\widehat{\mathbf{T}}$ is added to the interactional data (**IViDR-F**).
3. The case where only the residual part $\widetilde{\mathbf{T}}$ is added to the interactional data (**IViDR-R**).

### Table 4: The ablation study of our IViDR method on three real-world datasets

| Datasets | Coat NDCG@5 | Coat RECALL@5 | Yahoo!R3 NDCG@5 | Yahoo!R3 RECALL@5 | KuaiRand NDCG@5 | KuaiRand RECALL@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| IV4R-MF | $0.5617 \pm 0.0099$ | $0.5433 \pm 0.0097$ | $0.5653 \pm 0.0062$ | $0.7152 \pm 0.0085$ | $0.3741 \pm 0.0010$ | $0.3251 \pm 0.0008$ |
| iDCF | $0.5638 \pm 0.0124$ | $0.5493 \pm 0.0111$ | $0.6410 \pm 0.0022$ | $0.7780 \pm 0.0039$ | $0.4080 \pm 0.0004$ | $0.3481 \pm 0.0008$ |
| IViDR-T | $0.5669 \pm 0.0103$ | $0.5513 \pm 0.0106$ | $0.6428 \pm 0.0039$ | $0.7794 \pm 0.0036$ | $0.4104 \pm 0.0006$ | $0.3501 \pm 0.0007$ |
| IViDR-F | $0.5846 \pm 0.0121$ | $0.5710 \pm 0.0094$ | $0.6552 \pm 0.0041$ | $0.7849 \pm 0.0045$ | $0.4130 \pm 0.0005$ | $0.3529 \pm 0.0007$ |
| IViDR-R | $0.5875 \pm 0.0105$ | $0.5728 \pm 0.0101$ | $0.6514 \pm 0.0035$ | $0.7857 \pm 0.0030$ | $0.4151 \pm 0.0005$ | $0.3541 \pm 0.0008$ |
| **IViDR** | $\mathbf{0.5903 \pm 0.0101}$ | $\mathbf{0.5783 \pm 0.0114}$ | $\mathbf{0.6602 \pm 0.0038}$ | $\mathbf{0.7901 \pm 0.0037}$ | $\mathbf{0.4161 \pm 0.0004}$ | $\mathbf{0.3549 \pm 0.0006}$ |

A detailed analysis of the ablation studies can be found in Appendix G.1. We also performed a hyperparameter analysis, with a detailed discussion and complete results provided in Appendix G.2.

### 5.4 The Correctness of the Learned Latent Representation Using Our IViDR

We optimised the simulation data generation method from iDCF [37] to better match realistic scenarios. Specifically, we changed the economic level of users from a uniform distribution to a Gaussian-like distribution. We increased the number of users from 2,000 to 10,000, and the number of items from 300 to 1,000. Additionally, we added the mixing of item factors (see Appendix F).

There are three important hyperparameters: $\alpha$, which controls the density of the exposure vector; $\beta$, which represents the weight of the confounding effect on user preferences; and $\gamma$, which controls the weight of random noise in the user's exposure.

*Figure 3: (a) The visualisation of the true latent confounder. (b) The estimated latent confounders using iDCF. (c) The estimated latent confounders using IViDR.*

**The visualisation of the latent confounder.** Figure 3a shows the ground truth of the latent confounder with the exposure noise weight $\gamma = 0$. Figure 3b shows that the latent confounders predicted by iDCF still exhibit some overlap. Figure 3c demonstrates that the structure of the clusters of latent confounders estimated by IViDR is much closer to the true latent confounder distribution.

**Effect of Learning Identifiable Latent Confounders.** To accurately evaluate the difference between the estimated and true latent confounders, we calculated the mean correlation coefficient (MCC) between them. MCC is a widely used metric for evaluating the accuracy of estimated latent confounders [15]. For comparison, we adopted the same exposure noise settings (fixed $\alpha = 0.1$ and $\beta = 2.0$) as in iDCF [37]. The results, shown in Table 5, indicate that IViDR achieves more accurate estimations than iDCF, particularly as the exposure noise weight $\gamma$ increases.

### Table 5: Comparison of Models (MCC)

| Model | $\gamma = 0.0$ | $\gamma = 5.0$ | $\gamma = 10.0$ | $\gamma = 15.0$ | $\gamma = 20.0$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| iDCF | $0.8162$ | $0.7034$ | $0.6475$ | $0.5449$ | $0.4264$ |
| **IViDR (ours)** | $\mathbf{0.8405}$ | $\mathbf{0.7826}$ | $\mathbf{0.7659}$ | $\mathbf{0.6879}$ | $\mathbf{0.6262}$ |

---

## 6 Conclusion

In this paper, we propose a novel debiasing method (IViDR) that jointly integrates the IV method and iVAE for mitigating dual latent confounding biases in recommender systems. By leveraging the IV method and the iVAE, IViDR effectively mitigates dual confounding biases caused by latent confounders both between items and user feedback, and between item exposure and user feedback. This dual capability makes IViDR more robust compared to existing methods, which typically only address a single type of latent confounder. Our IViDR method uses user feature embeddings as IVs to reconstruct treatments and generate debiased interaction data, thereby mitigating confounding between item embeddings and user feedback. IViDR then uses iVAE to infer identifiable representations of latent confounders affecting item exposure and user feedback. We provided theoretical analyses demonstrating the soundness of using IVs and the identifiability of latent representations. Experimental evaluations on synthetic and three real-world datasets demonstrate that IViDR accurately estimates latent confounders and significantly reduces bias compared to existing methods, leading to improved recommendation performance.

---

## References

1. Himan Abdollahpouri, Masoud Mansoury, Robin Burke, and Bamshad Mobasher. 2019. The unfairness of popularity bias in recommendation. *arXiv preprint arXiv:1907.13286* (2019).
2. Gediminas Adomavicius and Alexander Tuzhilin. 2005. Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions. *IEEE transactions on knowledge and data engineering* 17, 6 (2005), 734–749.
3. Alexandre Belloni, Daniel Chen, Victor Chernozhukov, and Christian Hansen. 2012. Sparse models and methods for optimal instruments with an application to eminent domain. *Econometrica* 80, 6 (2012), 2369–2429.
4. Mehmet Caner and Bruce E Hansen. 2004. Instrumental variable estimation of a threshold model. *Econometric theory* 20, 5 (2004), 813–843.
5. Debo Cheng, Jiuyong Li, Lin Liu, Jixue Liu, and Thuc Duy Le. 2024. Data-driven causal effect estimation based on graphical causal modelling: A survey. *ACM Computing Surveys* 56, 5 (2024), 1–37.
6. Debo Cheng, Jiuyong Li, Lin Liu, Jiji Zhang, Jixue Liu, and Thuc Duy Le. 2022. Local search for efficient causal effect estimation. *IEEE Transactions on Knowledge and Data Engineering* 35, 9 (2022), 8823–8837.
7. Debo Cheng, Ziqi Xu, Jiuyong Li, Lin Liu, Jixue Liu, and Thuc Duy Le. 2023. Causal inference with conditional instruments using deep generative models. In *Proceedings of the AAAI Conference on Artificial Intelligence*, Vol. 37. 7122–7130.
8. Victor Chernozhukov, Guido W Imbens, and Whitney K Newey. 2007. Instrumental variable estimation of nonseparable models. *Journal of Econometrics* 139, 1 (2007), 4–14.
9. Sihao Ding, Peng Wu, Fuli Feng, Yitong Wang, Xiangnan He, Yong Liao, and Yongdong Zhang. 2022. Addressing Unmeasured Confounder for Recommendation with Sensitivity Analysis. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. ACM, 305–315.
10. Chongming Gao, Shijun Li, Yuan Zhang, Jiawei Chen, Biao Li, Wenqiang Lei, Peng Jiang, and Xiangnan He. 2022. Kuairand: an unbiased sequential recommendation dataset with randomly exposed videos. In *Proceedings of the 31st ACM International Conference on Information & Knowledge Management*. 3953–3957.
11. Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. *arXiv preprint arXiv:1703.04247* (2017).
12. Jason Hartford, Greg Lewis, Kevin Leyton-Brown, and Matt Taddy. 2017. Deep IV: A flexible approach for counterfactual prediction. In *International Conference on Machine Learning*. PMLR, 1414–1423.
13. Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In *Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval*. 639–648.
14. Guido W Imbens and Donald B Rubin. 2015. *Causal inference in statistics, social, and biomedical sciences*. Cambridge University Press.
15. Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvarinen. 2020. Variational autoencoders and nonlinear ica: A unifying framework. In *International Conference on Artificial Intelligence and Statistics*. PMLR, 2207–2217.
16. Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980* (2014).
17. Jan Kmenta. 2010. *Mostly harmless econometrics: An empiricist's companion*.
18. Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. *Computer* 42, 8 (2009), 30–37.
19. J. B. Kruskal. 1989. Rank, Decomposition, and Uniqueness for 3-Way and n-Way Arrays. *North-Holland Publishing Co.*, NLD, 7–18.
20. Haochen Liu, Da Tang, Ji Yang, Xiangyu Zhao, Hui Liu, Jiliang Tang, and Youlong Cheng. 2022. Rating Distribution Calibration for Selection Bias Mitigation in Recommendations. In *Proceedings of the ACM Web Conference 2022*. Association for Computing Machinery, 2048–2057.
21. Benjamin Marlin, Richard S Zemel, Sam Roweis, and Malcolm Slaney. 2012. Collaborative filtering and the missing at random assumption. *arXiv preprint arXiv:1206.5267* (2012).
22. Benjamin M Marlin and Richard S Zemel. 2009. Collaborative prediction and ranking with non-random missing data. In *Proceedings of the third ACM conference on Recommender systems*. 5–12.
23. Wang Miao, Wenjie Hu, Elizabeth L Ogburn, and Xiao-Hua Zhou. 2022. Identifying effects of multiple treatments in the presence of unmeasured confounding. *J. Amer. Statist. Assoc.* (2022), 1–15.
24. Judea Pearl. 2009. *Causality: Models, Reasoning and Inference* (2nd ed.). Cambridge University Press.
25. James Robins. 1986. A new approach to causal inference in mortality studies with a sustained exposure period—application to control of the healthy worker survivor effect. *Mathematical Modelling* 7, 9 (1986), 1393–1512.
26. James Robins. 1986. A new approach to causal inference in mortality studies with a sustained exposure period—application to control of the healthy worker survivor effect. *Mathematical Modelling* 7, 9 (1986), 1393–1512.
27. Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and Thorsten Joachims. 2016. Recommendations as treatments: Debiasing learning and evaluation. In *International Conference on Machine Learning*. PMLR, 1670–1679.
28. Zihua Si, Xueran Han, Xiao Zhang, Jun Xu, Yue Yin, Yang Song, and Ji-Rong Wen. 2022. A Model-Agnostic Causal Learning Framework for Recommendation using Search Data. In *Proceedings of the ACM Web Conference 2022*. 224–233.
29. Zihua Si, Zhongxiang Sun, Xiao Zhang, Jun Xu, Yang Song, Xiaoxue Zang, and Ji-Rong Wen. 2023. Enhancing recommendation with search data in a causal learning manner. *ACM Transactions on Information Systems* 41, 4 (2023), 1–31.
30. Arun Venkatraman, Wen Sun, Martial Hebert, J Bagnell, and Byron Boots. 2016. Online instrumental variable regression with applications to online linear system identification. In *Proceedings of the AAAI Conference on Artificial Intelligence*, Vol. 30.
31. Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & cross network for ad click predictions. In *Proceedings of the ADKDD'17*. 1–7.
32. Yixin Wang, Dawen Liang, Laurent Charlin, and David M Blei. 2020. Causal inference for recommender systems. In *Fourteenth ACM Conference on Recommender Systems*. 426–431.
33. Zimu Wang, Yue He, Jiashuo Liu, Wenchao Zou, Philip S Yu, and Peng Cui. 2022. Invariant Preference Learning for General Debiasing in Recommendation. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 1969–1978.
34. Le Wu, Xiangnan He, Xiang Wang, Kun Zhang, and Meng Wang. 2022. A survey on accuracy-oriented neural recommendation: From collaborative filtering to information-rich recommendation. *IEEE Transactions on Knowledge and Data Engineering* (2022).
35. Liyuan Xu, Yutian Chen, Siddarth Srinivasan, Nando de Freitas, Arnaud Doucet, and Arthur Gretton. 2020. Learning deep features in instrumental variable regression. *arXiv preprint arXiv:2010.07154* (2020).
36. Ruohan Zhan, Changhua Pei, Qiang Su, Jianfeng Wen, Xueliang Wang, Guanyu Mu, Dong Zheng, Peng Jiang, and Kun Gai. 2022. Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 4472–4481.
37. Qing Zhang, Xiaoying Zhang, Yang Liu, Hongning Wang, Min Gao, Jiheng Zhang, and Ruocheng Guo. 2023. Debiasing Recommendation by Learning Identifiable Latent Confounders. *arXiv preprint arXiv:2302.05052* (2023).
38. Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. 2019. Deep learning based recommender system: A survey and new perspectives. *ACM Computing Surveys (CSUR)* 52, 1 (2019), 1–38.
39. Yang Zhang, Fuli Feng, Xiangnan He, Tianxin Wei, Chonggang Song, Guohui Ling, and Yongdong Zhang. 2021. Causal Intervention for Leveraging Popularity Bias in Recommendation. In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*. 11–20.
40. Yu Zheng, Chen Gao, Xiang Li, Xiangnan He, Yong Li, and Depeng Jin. 2021. Disentangling user interest and conformity for recommendation with causal embedding. In *Proceedings of the Web Conference 2021*. 2980–2991.
41. Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. 1059–1068.
42. Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. 1059–1068.
43. Xinyuan Zhu, Yang Zhang, Fuli Feng, Xun Yang, Dingxian Wang, and Xiangnan He. 2022. Mitigating hidden confounding effects for causal recommendation. *arXiv preprint arXiv:2205.07499* (2022).
44. Yaochen Zhu, Jing Yi, Jiayi Xie, and Zhenzhong Chen. 2022. Deep causal reasoning for recommendations. *arXiv preprint arXiv:2201.02088* (2022).

---

## Appendix A Proof of Theorem 1

> **Theorem 3 (Restatement of Theorem 1).** *Given a causal DAG $\mathcal{G} = (\mathbf{Z} \cup \mathbf{W} \cup \mathbf{C} \cup \mathbf{B} \cup \{\mathbf{T}, \mathbf{A}, \mathbf{R}\}, \mathcal{E})$, where $\mathbf{T}$ and $\mathbf{R}$ are the treatment and outcome, respectively, and $\mathcal{E}$ is the set of edges between the variables. Let $\mathbf{B}$ denote the latent confounders between $\mathbf{T}$ and $\mathbf{R}$. Suppose there exists a directed edge $\mathbf{T} \to \mathbf{R}$ in $\mathcal{E}$, and $\mathbf{Z}$ represents the embeddings of user features. Additionally, assume that the underlying data generating process is faithful to the proposed causal DAG $\mathcal{G}$ as shown in Figure 1. Therefore, $\mathbf{Z}$ serves as the valid IV for estimating the causal effect of $\mathbf{T}$ on $\mathbf{R}$.*

*Proof.* First, we assume that the data-generating process is faithful to the causal DAG $\mathcal{G}$ in Figure 1, meaning that the conditional dependencies and independencies between the variables in the data can be accurately read from the DAG. Thus, we prove that the embeddings of user features, $\mathbf{Z}$, satisfy the three conditions of Definition 1 based on the causal DAG shown in Figure 1:
1. In the causal DAG $\mathcal{G}$, there exists a direct edge $\mathbf{Z} \to \mathbf{T}$, which implies that $\mathbf{Z} \not\!\perp\!\!\!\perp_d \mathbf{T}$. Therefore, the first condition of Definition 1 holds.
2. In the causal DAG $\mathcal{G}$, there are no direct paths from $\mathbf{Z}$ to $\mathbf{R}$ outside of $\mathbf{T}$, we have $\mathbf{Z} \perp\!\!\!\perp_d \mathbf{R} \mid \{\mathbf{T}, \mathbf{B}\}$. This satisfies the second condition of Definition 1.
3. Lastly, we establish that there is no confounding bias between $\mathbf{Z}$ and the outcome $\mathbf{R}$. In the causal DAG $\mathcal{G}$, there are no any back-door paths from $\mathbf{Z}$ to $\mathbf{R}$, i.e., $(\mathbf{Z} \perp\!\!\!\perp_d \mathbf{R})_{\mathcal{G}_{\underline{\mathbf{T}}}}$. Hence, the third condition of Definition 1 is satisfied.

Therefore, we conclude that $\mathbf{Z}$ satisfies all three conditions required for valid instrumental variables. $\square$

---

## Appendix B Algorithm

```text
Algorithm 1: Instrumental Variables-based Identifiable Disentangled Debiased Learning (IViDR).
Input: {X, i}, ∀i ∈ I; {A, W}, ∀u ∈ U; {r_ui}, ∀(u, i) ∈ D

// Treatment reconstruction using IVs
1: Reconstruct treatment T by instrumental variables Z to get the reconstructed treatment T^re.
2: Fuse the input X of iVAE with the reconstructed treatment T^re to obtain the debiased interactional data X^re.
3: X^re ← X + T_j^re

// Learning latent confounders
4: Calculate the latent confounders distribution q_ϕ(C_1 | A, W) for each user u by maximizing Eq. (40);
5: Calculate the latent confounders distribution q_ϕ(C_2 | A, W) for each user u by maximizing Eq. (40);
6: Fuse q_ϕ(C_1 | A, W) and q_ϕ(C_2 | A, W) to get latent confounders q_ϕ(C | A, W) (ρ, τ are tuning parameters).
7: C ← ρ * C_1 + τ * C_2

// Training recommendation model
8: Initialize a recommendation model f(u, i, C; η) with parameters η;
9: while Stop condition is not reached do
10:    Fetch (u, i) from D;
11:    Minimize the loss Eq. (25) to optimize η;
12: end
```

The algorithm consists of three main steps:
1. **Treatment reconstruction using IVs:** We use $\mathbf{Z}$ as the IV to reconstruct the treatment, obtaining $\mathbf{T}^{\text{re}}$. We then combine the input $\mathbf{X}$ of iVAE with $\mathbf{T}^{\text{re}}$ to obtain the debiased interactional data $\mathbf{X}^{\text{re}}$.
2. **Learning latent confounders:** We use the debiased interactional data $\mathbf{X}^{\text{re}}$ to calculate $q_\phi(\mathbf{C}_1 \mid \mathbf{A}, \mathbf{W})$ for each user $u$ by maximizing Eq. (40). We use the input $\mathbf{X}$ of iVAE to calculate $q_\phi(\mathbf{C}_2 \mid \mathbf{A}, \mathbf{W})$ for each user $u$ by maximizing Eq. (40). We then combine $q_\phi(\mathbf{C}_1 \mid \mathbf{A}, \mathbf{W})$ and $q_\phi(\mathbf{C}_2 \mid \mathbf{A}, \mathbf{W})$ to get latent confounder $q_\phi(\mathbf{C} \mid \mathbf{A}, \mathbf{W})$.
3. **Training the recommendation model:** We adjust for the learned latent representation to mitigate confounding bias.

---

## Appendix C Detailed Proofs of the Three Steps of Theorem 2

*Proof of Theorem 2.*

### Step I

We introduce here the volume of the matrix, denoted $\text{vol}\, M$, which is the product of the singular values of $M$. When $M$ has full column rank, $\text{vol}\, M = \sqrt{\det(M^T M)}$, and when $M$ is invertible, $\text{vol}\, M = |\det M|$. The matrix volume can substitute for Jacobi's absolute determinant in the variable transformation formula. This is especially useful when the Jacobian is a rectangular matrix ($n < d$). Suppose we have two sets of parameters: $(\mathbf{f}, \mathbf{H}, \boldsymbol{\lambda})$ and $(\widetilde{\mathbf{f}}, \widetilde{\mathbf{H}}, \widetilde{\boldsymbol{\lambda}})$ such that $p_{\mathbf{f}, \mathbf{H}, \boldsymbol{\lambda}}(\mathbf{X}^{\text{re}} \mid \mathbf{W}) = p_{\widetilde{\mathbf{f}}, \widetilde{\mathbf{H}}, \widetilde{\boldsymbol{\lambda}}}(\mathbf{X}^{\text{re}} \mid \mathbf{W})$ for all pairs $(\mathbf{X}^{\text{re}}, \mathbf{W})$. Then:
$$\int_c p_{\mathbf{H}, \boldsymbol{\lambda}}(\mathbf{C} \mid \mathbf{W}) p_{\mathbf{f}}(\mathbf{X}^{\text{re}} \mid \mathbf{C}) \, dc = \int_c p_{\widetilde{\mathbf{H}}, \widetilde{\boldsymbol{\lambda}}}(\mathbf{C} \mid \mathbf{W}) p_{\widetilde{\mathbf{f}}}(\mathbf{X}^{\text{re}} \mid \mathbf{C}) \, dc \tag{26}$$

According to Eq. (18), we get:
$$\int_c p_{\mathbf{H}, \boldsymbol{\lambda}}(\mathbf{C} \mid \mathbf{W}) p_{\boldsymbol{\varepsilon}}(\mathbf{X}^{\text{re}} - \mathbf{f}(\mathbf{C})) \, dc = \int_c p_{\widetilde{\mathbf{H}}, \widetilde{\boldsymbol{\lambda}}}(\mathbf{C} \mid \mathbf{W}) p_{\boldsymbol{\varepsilon}}(\mathbf{X}^{\text{re}} - \widetilde{\mathbf{f}}(\mathbf{C})) \, dc \tag{27}$$

In Eq. (27), we make variable substitutions $\overline{\mathbf{X}}^{\text{re}} = \mathbf{f}(\mathbf{C})$ on the left-hand side, $\overline{\mathbf{X}}^{\text{re}} = \widetilde{\mathbf{f}}(\mathbf{C})$ on the right-hand side, and $J$ denotes the Jacobian. We get:
$$\int_{\mathcal{X}} p_{\mathbf{H}, \boldsymbol{\lambda}}(\mathbf{f}^{-1}(\overline{\mathbf{X}}^{\text{re}}) \mid \mathbf{W}) \text{vol}\, J_{\mathbf{f}^{-1}}(\overline{\mathbf{X}}^{\text{re}}) p_{\boldsymbol{\varepsilon}}(\mathbf{X}^{\text{re}} - \overline{\mathbf{X}}^{\text{re}}) \, d\overline{\mathbf{X}}^{\text{re}} = \int_{\mathcal{X}} p_{\widetilde{\mathbf{H}}, \widetilde{\boldsymbol{\lambda}}}(\widetilde{\mathbf{f}}^{-1}(\overline{\mathbf{X}}^{\text{re}}) \mid \mathbf{W}) \text{vol}\, J_{\widetilde{\mathbf{f}}^{-1}}(\overline{\mathbf{X}}^{\text{re}}) p_{\boldsymbol{\varepsilon}}(\mathbf{X}^{\text{re}} - \overline{\mathbf{X}}^{\text{re}}) \, d\overline{\mathbf{X}}^{\text{re}} \tag{28}$$

In Eq. (28), we introduce:
$$\widetilde{p}_{\mathbf{H}, \boldsymbol{\lambda}, \mathbf{f}, \mathbf{W}}(\overline{\mathbf{X}}^{\text{re}}) = p_{\mathbf{H}, \boldsymbol{\lambda}}(\mathbf{f}^{-1}(\overline{\mathbf{X}}^{\text{re}}) \mid \mathbf{W}) \text{vol}\, J_{\mathbf{f}^{-1}}(\overline{\mathbf{X}}^{\text{re}}) \tag{29}$$
on both sides. We get:
$$\int_{\mathbb{R}^d} \widetilde{p}_{\mathbf{H}, \boldsymbol{\lambda}, \mathbf{f}, \mathbf{W}}(\overline{\mathbf{X}}^{\text{re}}) p_{\boldsymbol{\varepsilon}}(\mathbf{X}^{\text{re}} - \overline{\mathbf{X}}^{\text{re}}) \, d\overline{\mathbf{X}}^{\text{re}} = \int_{\mathbb{R}^d} \widetilde{p}_{\widetilde{\mathbf{H}}, \widetilde{\boldsymbol{\lambda}}, \widetilde{\mathbf{f}}, \mathbf{W}}(\overline{\mathbf{X}}^{\text{re}}) p_{\boldsymbol{\varepsilon}}(\mathbf{X}^{\text{re}} - \overline{\mathbf{X}}^{\text{re}}) \, d\overline{\mathbf{X}}^{\text{re}} \tag{30}$$

In Eq. (30), we use $*$ as the convolution operator:
$$(\widetilde{p}_{\mathbf{H}, \boldsymbol{\lambda}, \mathbf{f}, \mathbf{W}} * p_{\boldsymbol{\varepsilon}})(\mathbf{X}^{\text{re}}) = (\widetilde{p}_{\widetilde{\mathbf{H}}, \widetilde{\boldsymbol{\lambda}}, \widetilde{\mathbf{f}}, \mathbf{W}} * p_{\boldsymbol{\varepsilon}})(\mathbf{X}^{\text{re}}) \tag{31}$$

In Eq. (31), using $\mathcal{F}[\cdot]$ to designate the Fourier transform, and where $\varphi_{\boldsymbol{\varepsilon}} = \mathcal{F}[p_{\boldsymbol{\varepsilon}}]$:
$$\mathcal{F}[\widetilde{p}_{\mathbf{H}, \boldsymbol{\lambda}, \mathbf{f}, \mathbf{W}}](\boldsymbol{\omega}) \varphi_{\boldsymbol{\varepsilon}}(\boldsymbol{\omega}) = \mathcal{F}[\widetilde{p}_{\widetilde{\mathbf{H}}, \widetilde{\boldsymbol{\lambda}}, \widetilde{\mathbf{f}}, \mathbf{W}}](\boldsymbol{\omega}) \varphi_{\boldsymbol{\varepsilon}}(\boldsymbol{\omega}) \tag{32}$$

In Eq. (32), we remove $\varphi_{\boldsymbol{\varepsilon}}(\boldsymbol{\omega})$ from both sides, since it is non-zero almost everywhere (by assumption (i)):
$$\mathcal{F}[\widetilde{p}_{\mathbf{H}, \boldsymbol{\lambda}, \mathbf{f}, \mathbf{W}}](\boldsymbol{\omega}) = \mathcal{F}[\widetilde{p}_{\widetilde{\mathbf{H}}, \widetilde{\boldsymbol{\lambda}}, \widetilde{\mathbf{f}}, \mathbf{W}}](\boldsymbol{\omega}) \tag{33}$$

According to the inverse transformation of the Fourier transform, we get:
$$\widetilde{p}_{\mathbf{H}, \boldsymbol{\lambda}, \mathbf{f}, \mathbf{W}}(\mathbf{X}^{\text{re}}) = \widetilde{p}_{\widetilde{\mathbf{H}}, \widetilde{\boldsymbol{\lambda}}, \widetilde{\mathbf{f}}, \mathbf{W}}(\mathbf{X}^{\text{re}}) \tag{34}$$

### Step II

By taking the logarithm on both sides of Eq. (34) and replacing $p_{\mathbf{H}, \boldsymbol{\lambda}}$ by its expression from Eq. (19), we get:
$$\log \text{vol}\, J_{\mathbf{f}^{-1}}(\mathbf{X}^{\text{re}}) + \sum_{i=1}^n \left( \log Q_i(f_i^{-1}(\mathbf{X}^{\text{re}})) - \log Z_i(\mathbf{W}) + \sum_{j=1}^k H_{i,j}(f_i^{-1}(\mathbf{X}^{\text{re}})) \lambda_{i,j}(\mathbf{W}) \right) = \log \text{vol}\, J_{\widetilde{\mathbf{f}}^{-1}}(\mathbf{X}^{\text{re}}) + \sum_{i=1}^n \left( \log \widetilde{Q}_i(\widetilde{f}_i^{-1}(\mathbf{X}^{\text{re}})) - \log \widetilde{Z}_i(\mathbf{W}) + \sum_{j=1}^k \widetilde{H}_{i,j}(\widetilde{f}_i^{-1}(\mathbf{X}^{\text{re}})) \widetilde{\lambda}_{i,j}(\mathbf{W}) \right) \tag{35}$$

Let $\mathbf{W}_0, \dots, \mathbf{W}_{nk}$ be the points provided by assumption (iv), and define $\bar{\boldsymbol{\lambda}}(\mathbf{W}) = \boldsymbol{\lambda}(\mathbf{W}) - \boldsymbol{\lambda}(\mathbf{W}_0)$. We substitute each $\mathbf{W}_l$ into Eq. (35) to obtain $nk + 1$ such equations. We subtract the first equation of $\mathbf{W}_0$ from the remaining $nk$ equations to obtain the following for $l = 1, \dots, nk$:
$$\langle \mathbf{H}(\mathbf{f}^{-1}(\mathbf{X}^{\text{re}})), \bar{\boldsymbol{\lambda}}(\mathbf{W}_l) \rangle + \sum_i \log \frac{Z_i(\mathbf{W}_0)}{Z_i(\mathbf{W}_l)} = \langle \widetilde{\mathbf{H}}(\widetilde{\mathbf{f}}^{-1}(\mathbf{X}^{\text{re}})), \bar{\widetilde{\boldsymbol{\lambda}}}(\mathbf{W}_l) \rangle + \sum_i \log \frac{\widetilde{Z}_i(\mathbf{W}_0)}{\widetilde{Z}_i(\mathbf{W}_l)} \tag{36}$$

Let $L$ denote the matrix defined in assumption (iv), and $\widetilde{L}$ define $\bar{\widetilde{\boldsymbol{\lambda}}}$ in a similar way ($\widetilde{L}$ is not necessarily invertible). Define $s_l = \sum_i \log \frac{\widetilde{Z}_i(\mathbf{W}_0) Z_i(\mathbf{W}_l)}{Z_i(\mathbf{W}_0) \widetilde{Z}_i(\mathbf{W}_l)}$, and let $\mathbf{s}$ be the vector of all $s_l$, where $l = 1, \dots, nk$. Expressing Eq. (36) as a matrix form for all points $\mathbf{W}_l$, we get:
$$L^T \mathbf{H}(\mathbf{f}^{-1}(\mathbf{X}^{\text{re}})) = \widetilde{L}^T \widetilde{\mathbf{H}}(\widetilde{\mathbf{f}}^{-1}(\mathbf{X}^{\text{re}})) + \mathbf{s} \tag{37}$$

Then we multiply both sides of the above equations by $L^{-T}$:
$$\mathbf{H}(\mathbf{f}^{-1}(\mathbf{X}^{\text{re}})) = M \widetilde{\mathbf{H}}(\widetilde{\mathbf{f}}^{-1}(\mathbf{X}^{\text{re}})) + \mathbf{q} \tag{38}$$
where $M = L^{-T} \widetilde{L}^T$ and $\mathbf{q} = L^{-T} \mathbf{s}$.

To prove that $M$ is invertible, we first introduce Lemma 1:

> **Lemma 1.** *Consider a strong exponential distribution of size $k \ge 2$ with sufficient statistics $\mathbf{H}(\mathbf{X}^{\text{re}}) = (H_1(\mathbf{X}^{\text{re}}), \dots, H_k(\mathbf{X}^{\text{re}}))$. Suppose further that $\mathbf{H}$ is differentiable almost everywhere. Then there exist $k$ distinct values $\mathbf{X}_1^{\text{re}}$ through $\mathbf{X}_k^{\text{re}}$ such that $(\mathbf{H}'(\mathbf{X}_1^{\text{re}}), \dots, \mathbf{H}'(\mathbf{X}_k^{\text{re}}))$ is linearly independent in $\mathbb{R}^k$.*

*Proof of Lemma 1.* Suppose that for any chosen $k$ points, the family $(\mathbf{H}'(\mathbf{X}_1^{\text{re}}), \dots, \mathbf{H}'(\mathbf{X}_k^{\text{re}}))$ is never linearly independent. This means that $\mathbf{H}'(\mathbb{R})$ is contained in a subspace of $\mathbb{R}^k$ of dimension at most $k - 1$. Let $\boldsymbol{\theta}$ be a nonzero vector orthogonal to $\mathbf{H}'(\mathbb{R})$. Then for all $\mathbf{X}^{\text{re}} \in \mathbb{R}$, we have $\langle \mathbf{H}'(\mathbf{X}^{\text{re}}), \boldsymbol{\theta} \rangle = 0$. We find $\langle \mathbf{H}(\mathbf{X}^{\text{re}}), \boldsymbol{\theta} \rangle = \text{constant}$ by integration. Since this holds for all $\mathbf{X}^{\text{re}} \in \mathbb{R}$ and $\boldsymbol{\theta} \ne 0$, we conclude that the distribution is not strongly exponential, which contradicts Assumption 1. $\square$

### Step III

Now, by definition of $\mathbf{H}$ and assumption (iii), its Jacobian matrix exists and is a matrix of $nk \times n$ with rank $n$. This means that the Jacobian matrix of $\widetilde{\mathbf{H}} \circ \widetilde{\mathbf{f}}^{-1}$ exists and has a rank of $n$, and so does $M$. We distinguish two cases:
1. If $k = 1$, then this means that $M$ is invertible (because $M$ is $n \times n$).
2. If $k > 1$, we define $\mathbf{H}_i(X_i^{\text{re}}) = (H_{i,1}(X_i^{\text{re}}), \dots, H_{i,k}(X_i^{\text{re}}))$ and $\mathbf{X}^{\text{re}} = \mathbf{f}^{-1}(\mathbf{X}^{\text{re}})$. According to Lemma 1, for each $i \in [1, \dots, n]$, there exist $k$ points $X_{i1}^{\text{re}}, \dots, X_{ik}^{\text{re}}$ such that $(\mathbf{H}_i'(X_{i1}^{\text{re}}), \dots, \mathbf{H}_i'(X_{ik}^{\text{re}}))$ are linearly independent. Aggregate these points into $k$ vectors $\mathbf{X}_1^{\text{re}}, \dots, \mathbf{X}_k^{\text{re}}$, and splice the computed $k$ Jacobian matrices $J_{\mathbf{H}}(\mathbf{X}_l^{\text{re}})$ horizontally into a matrix:
$$Q = (J_{\mathbf{H}}(\mathbf{X}_1^{\text{re}}), \dots, J_{\mathbf{H}}(\mathbf{X}_k^{\text{re}}))$$
(also define $\widetilde{Q}$ as the Jacobian splice of $\widetilde{\mathbf{H}}(\widetilde{\mathbf{f}}^{-1} \circ \mathbf{f}(\mathbf{X}^{\text{re}}))$). Then the matrix $Q$ is invertible (by Lemma 1 and the combination of the fact that each component of $\widetilde{\mathbf{H}}$ is univariate). By deriving Eq. (38) for each $\mathbf{X}_l^{\text{re}}$, we get (in matrix form):
$$Q = M \widetilde{Q} \tag{39}$$

The invertibility of $Q$ implies the invertibility of $M$ and $\widetilde{Q}$. Thus, Eq. (38) and the invertibility of $M$ implies that $(\widetilde{\mathbf{f}}, \widetilde{\mathbf{H}}, \widetilde{\boldsymbol{\lambda}}) \sim (\mathbf{f}, \mathbf{H}, \boldsymbol{\lambda})$, thus completing the proof. $\square$

---

## Appendix D Identification of Latent Confounder Using iVAE

### D.1 The Detail of Obtaining the Approximate Posterior of Latent Confounders

Following the standard iVAE [15], we use $q_\phi(\mathbf{C} \mid \mathbf{A}, \mathbf{W})$ as the approximate posterior to learn $p_\theta(\mathbf{C} \mid \mathbf{A}, \mathbf{W})$:
$$\mathbb{E}[\log p_\theta(\mathbf{A} \mid \mathbf{W})] \ge \mathcal{L}(\theta, \phi) = \mathbb{E}\left[ \underbrace{\mathbb{E}_{q_\phi(\mathbf{C} \mid \mathbf{A}, \mathbf{W})} [\log p_\theta(\mathbf{C} \mid \mathbf{W}) - \log q_\phi(\mathbf{C} \mid \mathbf{A}, \mathbf{W})]}_{I} + \underbrace{\mathbb{E}_{q_\phi(\mathbf{C} \mid \mathbf{A}, \mathbf{W})} [\log p_\theta(\mathbf{A} \mid \mathbf{C})]}_{II} \right] \tag{40}$$

We can further decompose $\log p_\theta(\mathbf{A}, \mathbf{C} \mid \mathbf{W})$ into the following form:
$$\log p_\theta(\mathbf{A}, \mathbf{C} \mid \mathbf{W}) = \log p_\theta(\mathbf{A} \mid \mathbf{C}, \mathbf{W}) + \log p_\theta(\mathbf{C} \mid \mathbf{W}) = \log p_\theta(\mathbf{A} \mid \mathbf{C}) + \log p_\theta(\mathbf{C} \mid \mathbf{W}) \tag{41}$$

We employ the Gaussian distribution as the prior distribution for the representations, defined as:
$$p_\theta(\mathbf{C} \mid \mathbf{W}) := \mathcal{N}(\boldsymbol{\mu}_w(\mathbf{W}), \mathbf{v}_w(\mathbf{W})), \quad q_\phi(\mathbf{C} \mid \mathbf{A}, \mathbf{W}) := \mathcal{N}(\boldsymbol{\mu}_{aw}(\mathbf{A}, \mathbf{W}), \mathbf{v}_{aw}(\mathbf{A}, \mathbf{W})) \tag{42}$$
where $\mathcal{N}(\cdot, \cdot)$ denotes the Gaussian distribution. Thus, the computation of the expectation $I$ from Eq. (40) can be simplified to the following equation:
$$\mathbb{E}_{q_\phi(\mathbf{C} \mid \mathbf{A}, \mathbf{W})} [\log p_\theta(\mathbf{C} \mid \mathbf{W}) - \log q_\phi(\mathbf{C} \mid \mathbf{A}, \mathbf{W})] = -D_{\text{KL}}(\mathcal{N}(\boldsymbol{\mu}_{aw}(\mathbf{A}, \mathbf{W}), \mathbf{v}_{aw}(\mathbf{A}, \mathbf{W})) \parallel \mathcal{N}(\boldsymbol{\mu}_w(\mathbf{W}), \mathbf{v}_w(\mathbf{W}))) \tag{43}$$

For computing the $II$ part in Eq. (40), we have the following equation:
$$\log p_\theta(\mathbf{A} \mid \mathbf{C}) = \sum_{i=1}^n a_{ui} \log(\mu_c(\mathbf{C})_i) + (1 - a_{ui}) \log(1 - \mu_c(\mathbf{C})_i) \tag{44}$$

Thus, by maximizing Eq. (40), we can derive an approximate posterior $q_\phi(\mathbf{C} \mid \mathbf{A}, \mathbf{W})$ for recovering latent confounder $\mathbf{C}$.

---

## Appendix E Experimental Details

### E.1 Dataset Description

- **Coat Dataset:** The Coat dataset [27] simulates MNAR data from online coat purchases. It provides user features and ratings (5-point scale, where 1 indicates the lowest rating).
- **Yahoo!R3 Dataset:** The Yahoo!R3 dataset [22] contains user-item interactions from Yahoo!'s recommendation system. It also provides user features and ratings (5-point scale, where 1 indicates the lowest rating).
- **KuaiRand Dataset:** The KuaiRand dataset [10] includes 23,533 users and 6,712 videos. It also provides user features and signals for whether the user clicked (`IsClick=1` denoting a click).

---

## Appendix F Data Generation Process

The simulated dataset includes 10,000 users and 1,000 items. For each user $u$, $\mathbf{C}$ is represented as a 2D vector. This vector reflects the user's socio-economic status and is drawn from a mixture of five independent multivariate Gaussian distributions. For each item $i$, $\mathbf{V}$ is represented as a 2D vector reflecting the mixing of item factors, drawn from a mixture of five independent multivariate Gaussian distributions. The proxy variables $\mathbf{W}$ and $\mathbf{M}$ are one-dimensional categorical variables with probabilities determined by a Gaussian-like distribution. The embeddings of user features $\mathbf{Z}$ are randomly distributed.

The conditional distribution of $\mathbf{C}$ follows:
$$C_k \mid \mathbf{W} \sim \mathcal{N}(\mu_k(\mathbf{W}), \sigma_k^2(\mathbf{W})), \quad k \in \{1, 2\} \tag{45}$$
where $C_k$ is the $k$-th element of $\mathbf{C}$.

The conditional distribution of $\mathbf{V}$ follows:
$$V_k \mid \mathbf{M} \sim \mathcal{N}(\mu_k(\mathbf{M}), \sigma_k^2(\mathbf{M})), \quad k \in \{1, 2\} \tag{46}$$
where $V_k$ is the $k$-th element of $\mathbf{V}$.

We can reconstruct $\mathbf{C}$ with $\mathbf{V}$ to get the reconstructed latent confounders $\mathbf{H}$.

The $(u, i)$ is generated by:
$$a_{ui} \sim \text{Bernoulli}(g_i(\mathbf{H})), \quad g_i(\mathbf{H}) = \alpha \cdot \text{sigmoid}(\text{LeakyReLU}(\mathbf{H} M \mathbf{e}_{Hi}) + \gamma \epsilon) \tag{47}$$
where $M$ is a $2 \times 2$ matrix, $\mathbf{e}_{Hi}$ is a randomly generated item-wise 2D embedding vector, $\alpha$ is a hyperparameter, $\epsilon$ is random noise, and $\gamma$ is the corresponding weight of the noise.

The feedback is generated by:
$$r_{ui} = f_n(\mathbf{e}_u^T \mathbf{e}_i + \beta \mathbf{H}^T \mathbf{e}_{Hi} + \epsilon_{ui})$$
where $f_n : \mathbb{R} \to \{1, 2, 3, 4, 5\}$ is a normalization function, $\epsilon_{ui}$ is an i.i.d. random noise, and $\beta$ is a hyperparameter.

---

## Appendix G Ablation Study and Hyper-parameter Analysis

*Figure 4: The performance of all methods on the three real-world datasets.*

*Figure 5: IViDR-T, IViDR-F, IViDR-R, and IViDR algorithms' recommendation performance on the Coat, Yahoo!R3, and KuaiRand datasets.*

*Figure 6: Effect of the hyper-parameter selection: (a) Effect of the $\rho$ selection on NDCG@5 on the simulated datasets. (b) Effect of the $\rho$ selection on RECALL@5 on the simulated datasets. (c) Effect of the $\tau$ selection on NDCG@5 on the simulated datasets. (d) Effect of the $\tau$ selection on RECALL@5 on the simulated datasets.*

### G.1 Ablation Study

We conducted an ablation study to show the effectiveness of the proposed architecture:
1. **IV4R-MF:** IV4Rec applies IVs to decompose input vectors within recommendation models to mitigate the effects of latent confounders. IV4R-MF uses MF as the backbone.
2. **iDCF:** The iDCF incorporates proximal causal inference techniques to identifiably predict users' counterfactual feedback on items. It predicts more accurately than Deconfounder.
3. **IViDR-T:** The case where the original treatment $\mathbf{T}$ is added to the interactional data $\mathbf{X}$.
4. **IViDR-F:** The case where only fitted part $\widehat{\mathbf{T}}$ is added to the interactional data $\mathbf{X}$.
5. **IViDR-R:** The case where only residual part $\widetilde{\mathbf{T}}$ is added to the interactional data $\mathbf{X}$.

The experimental results are shown in Table 4:
- With the addition of the original treatment $\mathbf{T}$, IViDR-T slightly improved the iDCF results on all three real datasets, but the difference in results was small.
- With the addition of the fitted part $\widehat{\mathbf{T}}$, IViDR-F outperforms the iDCF results on all three real datasets. Specifically, for the Coat dataset, NDCG@5 improved from $0.5638$ to $0.5846$, and RECALL@5 improved from $0.5493$ to $0.5710$. For the Yahoo!R3 dataset, NDCG@5 improved from $0.6410$ to $0.6552$, and RECALL@5 improved from $0.7780$ to $0.7849$. For the KuaiRand dataset, NDCG@5 improved from $0.4080$ to $0.4130$, and RECALL@5 improved from $0.3481$ to $0.3529$.
- With the addition of the residual part $\widetilde{\mathbf{T}}$, IViDR outperforms IViDR-F on all three real datasets. Specifically, for the Coat dataset, NDCG@5 improved from $0.5846$ to $0.5903$, and RECALL@5 improved from $0.5710$ to $0.5783$. For the Yahoo!R3 dataset, NDCG@5 improved from $0.6552$ to $0.6602$, and RECALL@5 improved from $0.7849$ to $0.7901$. For the KuaiRand dataset, NDCG@5 improved from $0.4130$ to $0.4161$, and RECALL@5 improved from $0.3529$ to $0.3549$.

The recommendation performance of the IViDR-T, IViDR-F, IViDR-R, and IViDR algorithms on the Coat, Yahoo!R3, and KuaiRand datasets is shown in Figure 5.

### G.2 Hyper-parameter Analysis

We perform experiments to analyze the effect of the key hyperparameters of our method on the recommended performance.

Particularly, we fix all the other parameters (fixed $\alpha = 0.1$, $\beta = 2.0$, $\gamma = 0.0$), then set the $\rho$ and $\tau$ in Eq. (15) from $0$ to $1$ with an interval of $0.2$. As shown in Figure 6, we can see that:
1. As $\rho$ increases, NDCG@5 and RECALL@5 first get better and then decrease. This means that appropriate weighting of the debiased interactional data $\mathbf{X}^{\text{re}}$ can improve model recommendation performance.
2. As $\tau$ increases, NDCG@5 and RECALL@5 do not change much, but the general trend is still upward and then downward. This means that appropriate weighting of the interactional data $\mathbf{X}$ can also improve model recommendation performance.

