# Star profile: C1 / K=3 / trial_0052

Replaces `toy_large.ipynb` cell 58 ("CHUNK 14c STAR PROFILER") -- see `chunk14_star_profiler.py`'s module docstring for why the old version's core measure was retired, not just updated.

Memberships are reconstruction-weighted (each community column weighted by the reconstruction mass it carries); entities with no weight on any used column are listed as unassigned instead of ranked.

Unassigned live entities per facet: core_atom 6, core_child_he 2, cousin_he 8

`math_loss`=0.8449  `sociological_penalty`=0.0480  community_share (relation-level mass) sums to 1.000


## Community 0

- Relation-level mass share: **0.234**
- Domain mix (social share r_k): **0.472** (0.5 = even; caveat: FINDINGS §25, this measure does not track true domain balance on this corpus)
- Anchor-relation coherence: **1.000** (weighted 0 in sociological_penalty; shown for reference only, §4.17)

**Top articles (by weighted membership in this community):**

- 0.983 -- MG-BERT: leveraging unsupervised atomic representation learning for molecular property prediction
- 0.944 -- BioGPT: generative pre-trained transformer for biomedical text generation and mining
- 0.664 -- LM-GVP: an extensible sequence and structure informed deep learning framework for protein property prediction
- 0.024 -- LinkBERT: Pretraining Language Models with Document Links
- 0.020 -- The Diminishing Returns of Masked Language Models to Science
- 0.001 -- Protein Language Models and Structure Prediction: Connection and Progression
- 0.001 -- Generative Chemical Transformer: Neural Machine Learning of Molecular Geometric Structures from Chemical Language via Attention
- 0.001 -- An Empirical Study of Multi-Task Learning on BERT for Biomedical Text Mining
- 0.000 -- Generative Pre-Training from Molecules
- 0.000 -- PaLM: Scaling Language Modeling with Pathways

**Top authors (by weighted membership in this community):**

- 0.999 -- Ryan Brand
- 0.999 -- Panpan Xu
- 0.999 -- Miguel Romero Calvo
- 0.999 -- Emmanuel Oluwatobi Salawu
- 0.999 -- George Price
- 0.999 -- Sri Priya Ponnapalli
- 0.999 -- Colby J. Wise
- 0.999 -- Zichen Wang
- 0.769 -- Chang‐Yu Hsieh
- 0.653 -- Renqian Luo

**Top journals (by weighted membership in this community):**

- 0.996 -- Briefings in Bioinformatics
- 0.623 -- Scientific Reports
- 0.002 -- Journal of Chemical Information and Modeling
- 0.000 -- IEEE Transactions on Pattern Analysis and Machine Intelligence
- 0.000 -- Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)
- 0.000 -- International Conference on Learning Representations
- 0.000 -- npj Computational Materials
- 0.000 -- Research
- 0.000 -- ACM Transactions on Computing for Healthcare
- 0.000 -- Wireless Communications and Mobile Computing

**Top child hyperedges (by weighted membership in this community):**

- 0.990 -- (sibling_he study/C/en)
- 0.990 -- (sibling_he moreover/M/en)
- 0.965 -- (dummy_sibling study/P/en)
- 0.965 -- (sibling_he extensively/M/en)
- 0.948 -- (dummy_sibling integrate/P/en)
- 0.944 -- (sibling_he first/M/en one/C/en)
- 0.932 -- (sibling_he example/C/en)
- 0.909 -- (dummy_sibling make/P/en)
- 0.903 -- (dummy_sibling compare/P/en)
- 0.895 -- (sibling_he 540/M/en billion/M/en parameter/C/en)

**Top parent hyperedges (by weighted membership in this community):**

- 0.994 -- (parent (dummy_sibling propose/P/en) (focal_he bert/C/en bert/C/en graph/C/en mg/C/en molecular/M/en) (sibling_he study/C/en))
- 0.993 -- (parent (dummy_sibling leverage/P/en) (focal_he bert/C/en mg/C/en model/C/en) (sibling_he moreover/M/en) (sibling_he attention/C/en mechanism/C/en) (sibling_...
- 0.993 -- (parent (dummy_sibling generate/P/en) (focal_he bert/C/en mg/C/en model/C/en) (sibling_he can/M/en) (sibling_he atomic/M/en context/M/en representation/C/en ...
- 0.976 -- (parent (dummy_sibling study/P/en) (focal_he bert/C/en gpt/C/en i%2ee%2e/M/en variant/C/en variant/C/en) (sibling_he extensively/M/en) (sibling_he -/M/en bra...
- 0.976 -- (parent (dummy_sibling study/P/en) (focal_he -/M/en branch/C/en domain/C/en general/M/en language/C/en language_model/C/en main/M/en pre/M/en train/M/en two/...
- 0.966 -- (parent (dummy_sibling train/P/en) (focal_he language_model/C/en))
- 0.948 -- (parent (dummy_sibling integrate/P/en) (focal_he bert/C/en model/C/en powerful/M/en) (sibling_he gnns/C/en local/M/en mechanism/C/en network/C/en neural/M/en...
- 0.896 -- (parent (dummy_sibling not/M/en require/P/en) (focal_he bert/C/en mg/C/en model/C/en) (sibling_he craft/P/en feature/C/en hand/M/en input/C/en))
- 0.879 -- (parent (dummy_sibling pretraine/P/en) (focal_he bert/C/en mg/C/en model/C/en))
- 0.753 -- (parent (dummy_sibling pretrain/P/en) (focal_he amount/C/en bert/C/en datum/C/en large/M/en mg/C/en model/C/en unlabele/P/en) (sibling_he context/C/en inform...

**Top cousin hyperedges (by weighted membership in this community):**

- 1.000 -- (cousin_he field/C/en language/C/en natural/M/en nlp/C/en processing/C/en progress/C/en recent/M/en)
- 0.963 -- (cousin_he abundant/M/en biomedicine/C/en continual/M/en domain/C/en domain/C/en gain/C/en general/M/en language/C/en model/C/en pretraine/C/en result/C/en s...
- 0.952 -- (cousin_he biomedical/M/en domain/C/en)
- 0.889 -- (cousin_he \squad/M/en art/C/en benchmark/C/en glue/C/en new/M/en race/C/en result/C/en)
- 0.792 -- (cousin_he result/C/en)
- 0.774 -- (cousin_he corpus/C/en text/C/en)
- 0.758 -- (cousin_he biomedical/M/en mining/C/en representative/M/en task/C/en text/C/en three/M/en)
- 0.685 -- (cousin_he domain/C/en general/M/en great/M/en language/C/en natural/M/en success/C/en)
- 0.667 -- (cousin_he gain/C/en impressive/M/en language/C/en natural/M/en task/C/en)
- 0.644 -- (cousin_he language/C/en model/C/en palm/C/en pathway/C/en)

**Top fringe atoms (by weighted membership in this community):**

- 1.000 -- unlabele/M/en
- 1.000 -- abundant/M/en
- 1.000 -- substantial/M/en
- 1.000 -- biomedicine/C/en
- 1.000 -- continual/M/en
- 1.000 -- scratch/C/en
- 1.000 -- field/C/en
- 1.000 -- processing/C/en
- 1.000 -- progress/C/en
- 1.000 -- recent/M/en

## Community 1

- Relation-level mass share: **0.364**
- Domain mix (social share r_k): **0.438** (0.5 = even; caveat: FINDINGS §25, this measure does not track true domain balance on this corpus)
- Anchor-relation coherence: **1.000** (weighted 0 in sociological_penalty; shown for reference only, §4.17)

**Top articles (by weighted membership in this community):**

- 1.000 -- BioBERT: a pre-trained biomedical language representation model for biomedical text mining
- 1.000 -- Exploiting Pretrained Biochemical Language Models for Targeted Drug Design
- 0.933 -- ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning
- 0.889 -- Pushing the Boundaries of Molecular Property Prediction for Drug Discovery with Multitask Learning BERT Enhanced by SMILES Enumeration
- 0.600 -- An Empirical Study of Multi-Task Learning on BERT for Biomedical Text Mining
- 0.171 -- Generative Chemical Transformer: Neural Machine Learning of Molecular Geometric Structures from Chemical Language via Attention
- 0.116 -- The Diminishing Returns of Masked Language Models to Science
- 0.071 -- Protein Language Models and Structure Prediction: Connection and Progression
- 0.028 -- SMILES Transformer: Pre-trained Molecular Fingerprint for Low Data Drug Discovery
- 0.021 -- Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering

**Top authors (by weighted membership in this community):**

- 1.000 -- Donghyeon Kim
- 1.000 -- Jaewoo Kang
- 1.000 -- Jinhyuk Lee
- 1.000 -- Sungdong Kim
- 1.000 -- Chan Ho So
- 1.000 -- Sunkyu Kim
- 1.000 -- Wonjin Yoon
- 1.000 -- Nilgün Karalı
- 1.000 -- Gökçe Uludoğan
- 1.000 -- Kutlu Ö. Ülgen

**Top journals (by weighted membership in this community):**

- 1.000 -- Bioinformatics
- 0.926 -- IEEE Transactions on Pattern Analysis and Machine Intelligence
- 0.924 -- Research
- 0.003 -- Journal of Chemical Information and Modeling
- 0.000 -- Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)
- 0.000 -- International Conference on Learning Representations
- 0.000 -- Scientific Reports
- 0.000 -- npj Computational Materials
- 0.000 -- ACM Transactions on Computing for Healthcare
- 0.000 -- Wireless Communications and Mobile Computing

**Top core atoms (by weighted membership in this community):**

- 1.000 -- albert/C/en
- 1.000 -- ,/M/en
- 1.000 -- regressive/M/en
- 1.000 -- bfd/C/en
- 1.000 -- electra/C/en
- 1.000 -- xl/C/en
- 1.000 -- t5/C/en
- 1.000 -- uniref/C/en
- 1.000 -- auto/C/en
- 1.000 -- four/M/en

**Top child hyperedges (by weighted membership in this community):**

- 1.000 -- (focal_he ,/M/en albert/C/en auto/C/en auto/M/en bert/C/en bfd/C/en datum/C/en electra/C/en encoder/C/en four/M/en model/C/en model/C/en regressive/M/en t5/C...
- 0.964 -- (sibling_he datum/C/en gold/C/en mine/C/en vast/M/en)
- 0.937 -- (sibling_he smile/C/en transformer/C/en)
- 0.854 -- (focal_he bert/C/en bidirectional/M/en development/C/en encoder/C/en extraction/C/en information/C/en language/C/en model/C/en natural/M/en processing/C/en p...
- 0.852 -- (sibling_he advance/C/en recent/M/en)
- 0.850 -- (sibling_he amount/C/en datum/C/en large/M/en unlabele/P/en)
- 0.828 -- (focal_he bert/C/en large/C/en)
- 0.813 -- (dummy_sibling outperform/P/en)
- 0.781 -- (sibling_he largely/M/en)
- 0.778 -- (focal_he bert/C/en document/C/en method/C/en model/C/en single/M/en)

**Top parent hyperedges (by weighted membership in this community):**

- 1.000 -- (parent (dummy_sibling introduce/P/en) (focal_he -/M/en language_model/C/en pre/M/en train/M/en) (sibling_he recently/M/en))
- 0.984 -- (parent (dummy_sibling outperform/P/en) (focal_he art/C/en bert/C/en model/C/en previous/M/en) (sibling_he largely/M/en) (sibling_he biobert/C/en) (sibling_h...
- 0.980 -- (parent (dummy_sibling obtain/P/en) (focal_he bert/C/en) (sibling_he art/C/en comparable/M/en model/C/en performance/C/en previous/M/en))
- 0.940 -- (parent (dummy_sibling help/P/en) (focal_he -/M/en bert/C/en biomedical/M/en corpus/C/en pre/M/en training/M/en) (sibling_he biomedical/M/en complex/M/en tex...
- 0.935 -- (parent (dummy_sibling train/P/en use/P/en) (focal_he language_model/C/en) (sibling_he learn/P/en representation/C/en useful/M/en))
- 0.919 -- (parent (dummy_sibling name/P/en) (focal_he bert/C/en mtl/C/en) (sibling_he -/C/en large/M/en leverage/P/en pre/C/en scale/C/en training/C/en) (sibling_he le...
- 0.908 -- (parent (dummy_sibling leverage/P/en) (focal_he bert/C/en model/C/en mtl/C/en) (sibling_he additionally/M/en) (sibling_he attention/C/en mechanism/C/en) (sib...
- 0.871 -- (parent (dummy_sibling provide/P/en) (focal_he ideal/C/en language/C/en language_model/C/en lm/C/en natural/C/en nlp/C/en processing/C/en take/P/en) (sibling...
- 0.845 -- (parent (dummy_sibling train/P/en) (focal_he ,/M/en albert/C/en auto/C/en auto/M/en bert/C/en bfd/C/en datum/C/en electra/C/en encoder/C/en four/M/en model/C...
- 0.806 -- (parent (dummy_sibling exploit/P/en) (focal_he bert/C/en mtl/C/en) (sibling_he first/M/en) (sibling_he amount/C/en datum/C/en large/M/en unlabele/P/en) (sibl...

**Top cousin hyperedges (by weighted membership in this community):**

- 1.000 -- (cousin_he -/M/en -/M/en capability/C/en generalization/C/en great/M/en model/C/en our/M/en pre/M/en pre/M/en train/M/en training/M/en unsupervise/M/en)
- 1.000 -- (cousin_he capability/C/en transformer/M/en)
- 0.991 -- (cousin_he -/M/en pre/M/en)
- 0.914 -- (cousin_he impact/C/en learn/C/en our/M/en scale/C/en shoot/C/en understand/C/en)
- 0.901 -- (cousin_he -/M/en model/C/en semi/M/en supervise/M/en)
- 0.894 -- (cousin_he helpful/C/en not/M/en simply/M/en toxic/C/en untruthful/C/en user/C/en)
- 0.857 -- (cousin_he pre/C/en unsupervise/M/en)
- 0.814 -- (cousin_he good/M/en model/C/en our/M/en)
- 0.812 -- (cousin_he compute/C/en datum/C/en model/C/en size/C/en)
- 0.803 -- (cousin_he approach/C/en our/M/en)

**Top fringe atoms (by weighted membership in this community):**

- 1.000 -- training/M/en
- 1.000 -- generalization/C/en
- 1.000 -- train/M/en
- 1.000 -- pre/M/en
- 1.000 -- capability/C/en
- 1.000 -- -/M/en
- 1.000 -- unsupervise/M/en
- 1.000 -- transformer/M/en
- 0.973 -- understand/C/en
- 0.972 -- impact/C/en

## Community 2

- Relation-level mass share: **0.402**
- Domain mix (social share r_k): **0.524** (0.5 = even; caveat: FINDINGS §25, this measure does not track true domain balance on this corpus)
- Anchor-relation coherence: **1.000** (weighted 0 in sociological_penalty; shown for reference only, §4.17)

**Top articles (by weighted membership in this community):**

- 1.000 -- Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing
- 1.000 -- Mol‐BERT: An Effective Molecular Representation with BERT for Molecular Property Prediction
- 1.000 -- MatSciBERT: A materials domain language model for text mining and information extraction
- 1.000 -- ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators
- 1.000 -- Training language models to follow instructions with human feedback
- 1.000 -- ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
- 1.000 -- SMILES-BERT
- 1.000 -- LOGO, a contextualized pre-trained language model of human genome flexibly adapts to various downstream tasks by fine-tuning
- 1.000 -- A deep unsupervised language model for protein design
- 1.000 -- PaLM: Scaling Language Modeling with Pathways

**Top authors (by weighted membership in this community):**

- 1.000 -- Jianfeng Gao
- 1.000 -- Robert Tinn
- 1.000 -- Tristan Naumann
- 1.000 -- Hao Cheng
- 1.000 -- Xiaodong Liu
- 1.000 -- 裕二 池谷
- 1.000 -- Naoto Usuyama
- 1.000 -- Michael Lucas
- 1.000 -- Juncai Li
- 1.000 -- Mausam Mausam

**Top journals (by weighted membership in this community):**

- 1.000 -- Wireless Communications and Mobile Computing
- 1.000 -- ACM Transactions on Computing for Healthcare
- 1.000 -- npj Computational Materials
- 1.000 -- International Conference on Learning Representations
- 1.000 -- Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)
- 0.996 -- Journal of Chemical Information and Modeling
- 0.377 -- Scientific Reports
- 0.076 -- Research
- 0.074 -- IEEE Transactions on Pattern Analysis and Machine Intelligence
- 0.004 -- Briefings in Bioinformatics

**Top core atoms (by weighted membership in this community):**

- 1.000 -- question/C/en
- 1.000 -- lecture/C/en
- 1.000 -- chain/C/en
- 1.000 -- process/C/en
- 1.000 -- answer/P/en
- 1.000 -- multi/M/en
- 1.000 -- reasoning/C/en
- 1.000 -- mimic/P/en
- 1.000 -- think/C/en
- 1.000 -- hop/M/en

**Top child hyperedges (by weighted membership in this community):**

- 1.000 -- (sibling_he -/M/en answer/P/en chain/C/en cot/C/en explanation/C/en generate/P/en hop/M/en learn/P/en lecture/C/en mimic/P/en multi/M/en process/C/en questio...
- 1.000 -- (sibling_he %/C/en %/C/en 1%2e20/M/en 3%2e99/M/en cot/C/en fine/M/en gpt-3/C/en improve/P/en performance/C/en question/C/en shoot/C/en tune/P/en unifiedqa/C/...
- 1.000 -- (focal_he cot/C/en language_model/C/en utility/C/en)
- 1.000 -- (sibling_he evaluate/P/en generate/P/en molecule/C/en quality/C/en quantitatively/M/en)
- 0.999 -- (dummy_sibling corrupt/P/en)
- 0.999 -- (sibling_he may/M/en)
- 0.997 -- (sibling_he densely/M/en)
- 0.997 -- (dummy_sibling demonstrate/P/en)
- 0.997 -- (sibling_he scienceqa/C/en)
- 0.997 -- (dummy_sibling grow/P/en)

**Top parent hyperedges (by weighted membership in this community):**

- 1.000 -- (parent (dummy_sibling pretraine/P/en) (focal_he bert/C/en model/C/en))
- 1.000 -- (parent (dummy_sibling pretraine/P/en) (focal_he bert/C/en model/C/en) (sibling_he embed/C/en generate/P/en molecular/M/en substructure/C/en))
- 1.000 -- (parent (dummy_sibling pretraine/P/en) (focal_he language_model/C/en))
- 1.000 -- (parent (dummy_sibling discover/P/en) (focal_he bert/C/en complex/M/en entity/C/en model/C/en name/P/en recognition/C/en scheme/C/en tag/C/en unnecessary/C/e...
- 1.000 -- (parent (dummy_sibling pretraine/P/en) (focal_he bert/C/en language/C/en large/M/en model/C/en neural/M/en))
- 1.000 -- (parent (dummy_sibling scibert/C/en) (focal_he scibert/C/en) (sibling_he matscibert/C/en outperform/C/en) (sibling_he language_model/C/en))
- 1.000 -- (parent (dummy_sibling base/P/en contextualize/P/en pre/P/en) (focal_he -/M/en language_model/C/en train/M/en))
- 1.000 -- (parent (dummy_sibling develop/P/en) (focal_he language_model/C/en protgpt2/C/en) (sibling_he base/P/en evident/M/en generative/M/en gpt/C/en language_model/...
- 1.000 -- (parent (dummy_sibling increase/P/en) (focal_he bert/C/en speed/C/en training/C/en))
- 1.000 -- (parent (dummy_sibling align/P/en) (focal_he language_model/C/en))

**Top cousin hyperedges (by weighted membership in this community):**

- 1.000 -- (dummy_cousin name/P/en)
- 1.000 -- (cousin_he can/M/en)
- 1.000 -- (dummy_cousin develop/P/en)
- 1.000 -- (dummy_cousin present/P/en)
- 1.000 -- (cousin_he paper/C/en)
- 1.000 -- (cousin_he effective/M/en)
- 1.000 -- (dummy_cousin investigate/P/en)
- 1.000 -- (dummy_cousin fine‐tune/P/en)
- 1.000 -- (dummy_cousin combine/P/en)
- 1.000 -- (dummy_cousin tailor/P/en)

**Top fringe atoms (by weighted membership in this community):**

- 1.000 -- property/C/en
- 1.000 -- molecular/M/en
- 1.000 -- prediction/C/en
- 1.000 -- name/P/en
- 1.000 -- fine‐tune/P/en
- 1.000 -- combine/P/en
- 1.000 -- tailor/P/en
- 1.000 -- extract/P/en
- 1.000 -- develop/P/en
- 1.000 -- present/P/en