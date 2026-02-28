# SL-IBPM
**Title:** Sources Localization via Information Back Propagation Mechanism

**Abstract:** With the rapid expansion of social networks, the rampant spread of rumors and misinformation has increasingly threatened social stability, making timely and accurate identification of information sources an urgent need. Traditional source localization approaches struggle with the inherent randomness in information propagation, while existing deep learning methods often fail to balance prediction accuracy and model complexity or rely excessively on large-scale observational data. To address these challenges, this paper proposes a novel deep learning framework that enables effective Source Localization through an innovative Information Back Propagation Mechanism (SL-IBPM). Derived from this mechanism, SL-IBPM encapsulates two core components: self-information Damping and neighbor-information Aggregation. These components are integrated into the BackProp module, which can be embedded into various unidirectional propagation models. This design achieves a favorable balance between prediction accuracy and computational efficiency and enhances adaptability to different propagation scenarios. Additionally, a weighted loss function is employed to mitigate the class imbalance issue, where non-source nodes far outnumber source nodes. Extensive experiments on six real-world networks and two real propagation cascade datasets show that SL-IBPM outperforms state-of-the-art methods in source detection accuracy. It maintains low model complexity and exhibits robust performance even with limited observational data, highlighting its practical superiority.

**Dataset:** [Zenodo Dataset URL](https://zenodo.org/records/18491212).

**How to run?** After downloading and extracting the dataset, place it in the same folder, then simply run the corresponding `.py` file.

**SL-IBPM Model:**

``
python ibpm.py
``
