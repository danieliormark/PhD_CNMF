# G3 hand-read sample, trial slices 2020/2021 (30 units, seed 107)

Drawn from 6,855 units of 463 trial articles. For each unit: the sentence, then each parent (one per focal occurrence) with its children, then the cousins (other clauses) with the parents they attach to. After each hyperedge: atom ← source words.

## 1. PMC9166894.r1.L129.S7.U1

> Sampling from a transformer model trained in this way would yield a closer approximation of Eq. 1.

- **parent** P1
  - `(dummy_sibling train/P/en)` — train←'trained'
  - `(focal_he transformer_model/C/focal)` — transformer_model←'model transformer'
  - `(sibling_he way/C/en)` — way←'way'
- cousin `(dummy_cousin would/M/en yield/P/en)` → P1 — would←'would' yield←'yield'
- cousin `(cousin_he approximation/C/en close/M/en)` → P1 — approximation←'approximation' close←'closer'

Reading: 

## 2. PMC7597339.r1.L61.S2.U1

> Due to the lack of text_labelled data in the field of cultural relics, this paper utilizes BERT to complete the knowledge graph of cultural relics to address the problem of the lack of high_quality labelled data in the field of cultural relics.

- **parent** P1
  - `(dummy_sibling utilize/P/en)` — utilize←'utilizes'
  - `(focal_he bert/C/focal)` — bert←'BERT'
  - `(sibling_he cultural/M/en datum/C/en field/C/en lack/C/en relic/C/en text_labelled/M/en)` — cultural←'cultural cultural cultural' datum←'data data' field←'field field' lack←'lack lack' relic←'relics relics relics' text_labelled←'text_labelled'
  - `(sibling_he paper/C/en)` — paper←'paper'
  - `(sibling_he complete/P/en cultural/M/en graph/C/en knowledge/C/en relic/C/en)` — complete←'complete' cultural←'cultural cultural cultural' graph←'graph' knowledge←'knowledge' relic←'relics relics relics'
  - `(sibling_he address/P/en cultural/M/en datum/C/en field/C/en high_quality/M/en label/P/en lack/C/en problem/C/en relic/C/en)` — address←'address' cultural←'cultural cultural cultural' datum←'data data' field←'field field' high_quality←'high_quality' label←'labelled' lack←'lack lack' problem←'problem' relic←'relics relics relics'

Reading: 

## 3. PMC8011068.r1.L102.S5.U1

> Beyond the capability to generate coherent texts, Transformer models have the potential to perform other tasks such as question answering and document classification with a limited number of training examples.

- **parent** P1
  - `(dummy_sibling have/P/en)` — have←'have'
  - `(focal_he transformer_model/C/focal)` — transformer_model←'Transformer models'
  - `(sibling_he potential/C/en)` — potential←'potential'
  - `(sibling_he answering/C/en classification/C/en document/C/en example/C/en limited/M/en number/C/en perform/P/en question/C/en task/C/en training/C/en)` — answering←'answering' classification←'classification' document←'document' example←'examples' limited←'limited' number←'number' perform←'perform' question←'question' task←'tasks' training←'training'
- cousin `(dummy_cousin generate/P/en)` → P1 — generate←'generate'
- cousin `(cousin_he coherent/M/en text/C/en)` → P1 — coherent←'coherent' text←'texts'
- cousin `(cousin_he capability/C/en)` → P1 — capability←'capability'

Reading: 

## 4. PMC8380587.r1.L94.S1.U1

> The research was mainly divided into 3 stages: The first stage was fine_tuning the pretrained language model BERT, the second stage was to transform the upper structure of the language model obtained in the first stage to obtain a text classification model and then fine_tune the classification model, and the third stage was to ensemble the classification model obtained in the second stage, so we could get the best model combination, and then introduce “pseudo data” to retrain the best combination models to improve the overall classification results.

- **parent** P1
  - `(dummy_sibling fine_tune/P/en)` — fine_tune←'fine_tuning'
  - `(focal_he bert/C/focal language_model/C/focal pretrained/M/en)` — bert←'BERT' language_model←'language language model model' pretrained←'pretrained'
  - `(sibling_he first/M/en stage/C/en)` — first←'first first' stage←'stage stage stage stage stage'
- **parent** P2
  - `(dummy_sibling transform/P/en)` — transform←'transform'
  - `(focal_he first/M/en language_model/C/focal obtain/P/en stage/C/en structure/C/en upper/M/en)` — first←'first first' language_model←'language language model model' obtain←'obtained obtained' stage←'stage stage stage stage stage' structure←'structure' upper←'upper'
  - `(sibling_he classification/C/en model/C/en obtain/P/en text/C/en)` — classification←'classification classification classification classification' model←'model model model model model model' obtain←'obtain' text←'text'
- cousin `(dummy_cousin divide/P/en)` → P1, P2 — divide←'divided'
- cousin `(cousin_he research/C/en)` → P1, P2 — research←'research'
- cousin `(cousin_he 3/M/en stage/C/en)` → P1, P2 — 3←'3' stage←'stages'
- cousin `(dummy_cousin be/P/en)` → P1, P2 — be←'was was'
- cousin `(cousin_he second/M/en stage/C/en)` → P1, P2 — second←'second second' stage←'stage stage stage stage stage'
- cousin `(dummy_cousin transform/P/en)` → P1 — transform←'transform'
- cousin `(cousin_he structure/C/en upper/M/en)` → P1 — structure←'structure' upper←'upper'
- cousin `(dummy_cousin obtain/P/en)` → P1, P2 — obtain←'obtained obtained'
- cousin `(cousin_he first/M/en stage/C/en)` → P1, P2 — first←'first first' stage←'stage stage stage stage stage'
- cousin `(cousin_he classification/C/en model/C/en text/C/en)` → P1 — classification←'classification classification classification classification' model←'model model model model model model' text←'text'
- cousin `(dummy_cousin fine_tune/P/en)` → P1, P2 — fine_tune←'fine_tune'
- cousin `(cousin_he classification/C/en model/C/en)` → P1, P2 — classification←'classification classification classification classification' model←'model model model model model model'
- cousin `(cousin_he stage/C/en third/M/en)` → P1, P2 — stage←'stage stage stage stage stage' third←'third'
- cousin `(dummy_cousin ensemble/P/en)` → P1, P2 — ensemble←'ensemble'
- cousin `(dummy_cousin could/M/en get/P/en)` → P1, P2 — could←'could' get←'get'
- cousin `(cousin_he combination/C/en good/M/en model/C/en)` → P1, P2 — combination←'combination combination' good←'best best' model←'model model model model model model'
- cousin `(dummy_cousin introduce/P/en)` → P1, P2 — introduce←'introduce'
- cousin `(cousin_he datum/C/en pseudo/M/en)` → P1, P2 — datum←'data' pseudo←'pseudo'
- cousin `(dummy_cousin retrain/P/en)` → P1, P2 — retrain←'retrain'
- cousin `(dummy_cousin improve/P/en)` → P1, P2 — improve←'improve'
- cousin `(cousin_he classification/C/en overall/M/en result/C/en)` → P1, P2 — classification←'classification classification classification classification' overall←'overall' result←'results'
- cousin `(cousin_he pretrained/M/en)` → P2 — pretrained←'pretrained'

Reading: 

## 5. PMC7889424.r1.L251.S2.U1

> Especially for the test set, this indicates that the combination of the different heads improves BERT’s performance.

- **parent** P1
  - `(dummy_sibling improve/P/en)` — improve←'improves'
  - `(focal_he bert/C/focal performance/C/en)` — bert←'BERT' performance←'performance'
  - `(sibling_he combination/C/en different/M/en head/C/en)` — combination←'combination' different←'different' head←'heads'
- cousin `(dummy_cousin indicate/P/en)` → P1 — indicate←'indicates'
- cousin `(cousin_he set/C/en test/C/en)` → P1 — set←'set' test←'test'

Reading: 

## 6. PMC8129876.r1.L232.S1.U1

> To assess the utility of our classification approaches and gain an understanding of the data, we used the best_performing classifier (the classifier based on BERT) to label all collected unlabeled data and compute the data distribution.

- **parent** P1
  - `(dummy_sibling base/P/en)` — base←'based'
  - `(focal_he bert/C/focal)` — bert←'BERT'
  - `(sibling_he collect/P/en compute/P/en datum/C/en distribution/C/en label/P/en unlabeled/M/en)` — collect←'collected' compute←'compute' datum←'data data data' distribution←'distribution' label←'label' unlabeled←'unlabeled'
- cousin `(dummy_cousin use/P/en)` → P1 — use←'used'
- cousin `(dummy_cousin assess/P/en)` → P1 — assess←'assess'
- cousin `(cousin_he approach/C/en classification/C/en our/M/en utility/C/en)` → P1 — approach←'approaches' classification←'classification' our←'our' utility←'utility'
- cousin `(dummy_cousin gain/P/en)` → P1 — gain←'gain'
- cousin `(cousin_he datum/C/en understanding/C/en)` → P1 — datum←'data data data' understanding←'understanding'
- cousin `(cousin_he best_performing/M/en classifier/C/en)` → P1 — best_performing←'best_performing' classifier←'classifier classifier'

Reading: 

## 7. PMC8981604.r1.L179.S3.U1

> We leave the study that compares BERT models fine_tuned on Chinese clinical texts for future work.

- **parent** P1
  - `(dummy_sibling compare/P/en)` — compare←'compares'
  - `(focal_he bert/C/focal model/C/en)` — bert←'BERT' model←'models'
- cousin `(cousin_he future/M/en work/C/en)` → P1 — future←'future' work←'work'
- cousin `(dummy_cousin leave/P/en)` → P1 — leave←'leave'
- cousin `(cousin_he study/C/en)` → P1 — study←'study'
- cousin `(dummy_cousin fine_tune/P/en)` → P1 — fine_tune←'fine_tuned'
- cousin `(cousin_he chinese/M/en clinical/M/en text/C/en)` → P1 — chinese←'Chinese' clinical←'clinical' text←'texts'

Reading: 

## 8. PMC8137882.r1.L55.S5.U1

> BERT has also been embraced by the clinical domain.

- **parent** P1
  - `(dummy_sibling be/M/en embrace/P/en)` — be←'been' embrace←'embraced'
  - `(focal_he bert/C/focal)` — bert←'BERT'
  - `(sibling_he clinical/M/en domain/C/en)` — clinical←'clinical' domain←'domain'

Reading: 

## 9. PMC7566510.r1.L244.S3.U1

> Applying the same BERT model on top of Lucene(a + b + c + d + e) yields the best accuracy we achieved, 83.56%.

- **parent** P1
  - `(dummy_sibling apply/P/en)` — apply←'Applying'
  - `(focal_he bert/C/focal model/C/en)` — bert←'BERT' model←'model'
  - `(sibling_he lucene%28a/C/en top/C/en)` — lucene%28a←'Lucene(a' top←'top'
- cousin `(dummy_cousin achieve/P/en yield/P/en)` → P1 — achieve←'achieved' yield←'yields'
- cousin `(cousin_he accuracy/C/en good/M/en)` → P1 — accuracy←'accuracy' good←'best'
- cousin `(cousin_he 83%2e56/M/en percent/C/en)` → P1 — 83%2e56←'83.56' percent←'%'

Reading: 

## 10. PMC8339965.r1.L149.S3.U1

> Table 4 reports the accuracy of the baseline classifiers and our proposed BERT-based classifiers on each of the four subtests, separately trained and tested on clean and verbatim transcripts.

- **parent** P1
  - `(dummy_sibling propose/P/en)` — propose←'proposed'
  - `(focal_he based/M/en bert/C/focal classifier/C/en)` — based←'based' bert←'BERT' classifier←'classifiers classifiers'
- cousin `(dummy_cousin report/P/en)` → P1 — report←'reports'
- cousin `(cousin_he accuracy/C/en baseline/C/en classifier/C/en our/M/en)` → P1 — accuracy←'accuracy' baseline←'baseline' classifier←'classifiers classifiers' our←'our'
- cousin `(dummy_cousin separately/M/en test/P/en train/P/en)` → P1 — separately←'separately' test←'tested' train←'trained'
- cousin `(cousin_he clean/M/en transcript/C/en verbatim/C/en)` → P1 — clean←'clean' transcript←'transcripts' verbatim←'verbatim'
- cousin `(cousin_he four/M/en subtest/C/en)` → P1 — four←'four' subtest←'subtests'

Reading: 

## 11. PMC8665397.r1.L550.S3.U1

> Indeed, a language model trained only on the form (raw text) cannot capture the true meaning of the text.

- **parent** P1
  - `(dummy_sibling can/M/en capture/P/en not/M/en)` — can←'can' capture←'capture' not←'not'
  - `(focal_he language_model/C/focal)` — language_model←'language model'
  - `(sibling_he form/C/en only/M/en raw/M/en text/C/en train/P/en)` — form←'form' only←'only' raw←'raw' text←'text text' train←'trained'
  - `(sibling_he meaning/C/en text/C/en true/M/en)` — meaning←'meaning' text←'text text' true←'true'

Reading: 

## 12. PMC8981604.r1.L175.S1.U1

> Exemplified by BERT (Bidirectional Encoder Representations from Transformers), contextual word representations encode each word using not only the distributed vector of the word itself, but also distributed vectors of surrounding words that have semantic dependencies with the word.

- **parent** P1
  - `(dummy_sibling exemplify/P/en)` — exemplify←'Exemplified'
  - `(focal_he bert/C/focal)` — bert←'BERT Bidirectional Encoder Representations Transformers from'
- cousin `(dummy_cousin encode/P/en)` → P1 — encode←'encode'
- cousin `(cousin_he contextual/M/en representations/C/en word/C/en)` → P1 — contextual←'contextual' representations←'representations' word←'word word word word'
- cousin `(cousin_he word/C/en)` → P1 — word←'word word word word'
- cousin `(dummy_cousin use/P/en)` → P1 — use←'using'
- cousin `(dummy_cousin distribute/P/en)` → P1 — distribute←'distributed distributed'
- cousin `(cousin_he vector/C/en)` → P1 — vector←'vector'
- cousin `(dummy_cousin surround/P/en)` → P1 — surround←'surrounding'
- cousin `(dummy_cousin have/P/en)` → P1 — have←'have'
- cousin `(cousin_he dependency/C/en semantic/M/en word/C/en)` → P1 — dependency←'dependencies' semantic←'semantic' word←'word word word word'

Reading: 

## 13. PMC7732709.r1.L285.S1.U1

> A total of 3 learning algorithms for regression were used for ensembling language model outputs and features: linear regression, Bayesian regression, and ridge regression.

- **parent** P1
  - `(dummy_sibling ensemble/P/en)` — ensemble←'ensembling'
  - `(focal_he feature/C/en language_model/C/focal output/C/en)` — feature←'features' language_model←'language model' output←'outputs'
- cousin `(dummy_cousin be/M/en use/P/en)` → P1 — be←'were' use←'used'
- cousin `(cousin_he 3/M/en algorithm/C/en learning/C/en regression/C/en total/C/en)` → P1 — 3←'3' algorithm←'algorithms' learning←'learning' regression←'regression regression regression regression' total←'total'
- cousin `(cousin_he bayesian/M/en linear/M/en regression/C/en ridge/C/en)` → P1 — bayesian←'Bayesian' linear←'linear' regression←'regression regression regression regression' ridge←'ridge'

Reading: 

## 14. PMC8001296.r1.L133.S2.U1

> In BERT pre_training, the positive samples are selected from the same document and the negative samples are chosen in documents with different topics, so it is quite easy for the neural network model to distinguish them.

- **parent** P1
  - `(dummy_sibling be/M/en select/P/en)` — be←'are are' select←'selected'
  - `(focal_he bert/C/focal pre_training/C/en)` — bert←'BERT' pre_training←'pre_training'
  - `(sibling_he positive/M/en sample/C/en)` — positive←'positive' sample←'samples samples'
  - `(sibling_he document/C/en)` — document←'document'
- cousin `(dummy_cousin be/M/en choose/P/en)` → P1 — be←'are are' choose←'chosen'
- cousin `(cousin_he negative/M/en sample/C/en)` → P1 — negative←'negative' sample←'samples samples'
- cousin `(cousin_he different/M/en document/C/en topic/C/en)` → P1 — different←'different' document←'documents' topic←'topics'
- cousin `(dummy_cousin be/P/en)` → P1 — be←'is'
- cousin `(cousin_he easy/C/en quite/M/en)` → P1 — easy←'easy' quite←'quite'
- cousin `(dummy_cousin distinguish/P/en)` → P1 — distinguish←'distinguish'
- cousin `(cousin_he model/C/en network/C/en neural/M/en)` → P1 — model←'model' network←'network' neural←'neural'

Reading: 

## 15. PMC8238198.r1.L377.S4.U1

> Note that AUBER outperforms all of AUBER's competitors on regularizing BERT, providing the best performance for all the datasets.

- **parent** P1
  - `(dummy_sibling regularize/P/en)` — regularize←'regularizing'
  - `(focal_he bert/C/focal)` — bert←'BERT'
- cousin `(dummy_cousin note/P/en)` → P1 — note←'Note'
- cousin `(dummy_cousin outperform/P/en)` → P1 — outperform←'outperforms'
- cousin `(cousin_he auber/C/en)` → P1 — auber←'AUBER AUBER'
- cousin `(cousin_he auber/C/en competitor/C/en)` → P1 — auber←'AUBER AUBER' competitor←'competitors'
- cousin `(dummy_cousin provide/P/en)` → P1 — provide←'providing'
- cousin `(cousin_he good/M/en performance/C/en)` → P1 — good←'best' performance←'performance'
- cousin `(cousin_he dataset/C/en)` → P1 — dataset←'datasets'

Reading: 

## 16. PMC9166894.r1.L288.S6.U1

> In order to use just_in_time AIG in conjunction with CAT, it is imperative that the item generating method—in our case language models —reliably produce items that represent a requested construct, i.e., hold validity, without exceptions.

- **parent** P1
  - `(dummy_sibling be/P/en)` — be←'is'
  - `(focal_he case/C/en generate/P/en item/C/en language_model/C/focal method/C/en our/M/en)` — case←'case' generate←'generating' item←'item' language_model←'language models' method←'method' our←'our'
  - `(sibling_he imperative/C/en)` — imperative←'imperative'
  - `(sibling_he construct/C/en exception/C/en item/C/en produce/P/en reliably/M/en represent/P/en request/P/en validity/C/en)` — construct←'construct' exception←'exceptions' item←'items' produce←'produce' reliably←'reliably' represent←'represent' request←'requested' validity←'validity'
- cousin `(dummy_cousin use/P/en)` → P1 — use←'use'
- cousin `(cousin_he just_in_time/C/en)` → P1 — just_in_time←'just_in_time'
- cousin `(cousin_he aig/C/en)` → P1 — aig←'AIG'
- cousin `(cousin_he cat/C/en conjunction/C/en)` → P1 — cat←'CAT' conjunction←'conjunction'
- cousin `(cousin_he order/C/en)` → P1 — order←'order'

Reading: 

## 17. PMC7544600.r1.L124.S2.U1

> Finally, explored cross_lingual transfer for Danish using several architectures for supervised NER, including Flair, fastText, BPE and both monolingual (Danish) and multilingual BERT, on a modestly_sized training set, testing different training and fine_tuning approaches.

- **parent** P1
  - `(dummy_sibling include/P/en)` — include←'including'
  - `(focal_he bert/C/focal both/M/en bpe/C/en danish/M/en fasttext/C/en flair/C/en monolingual/M/en multilingual/C/en)` — bert←'BERT' both←'both' bpe←'BPE' danish←'Danish' fasttext←'fastText' flair←'Flair' monolingual←'monolingual' multilingual←'multilingual'
- cousin `(dummy_cousin explore/P/en)` → P1 — explore←'explored'
- cousin `(cousin_he cross_lingual/M/en danish/C/en transfer/C/en)` → P1 — cross_lingual←'cross_lingual' danish←'Danish' transfer←'transfer'
- cousin `(dummy_cousin use/P/en)` → P1 — use←'using'
- cousin `(cousin_he architecture/C/en)` → P1 — architecture←'architectures'
- cousin `(dummy_cousin supervise/P/en)` → P1 — supervise←'supervised'
- cousin `(cousin_he ner/C/en)` → P1 — ner←'NER'
- cousin `(cousin_he modestly_sized/M/en set/C/en training/C/en)` → P1 — modestly_sized←'modestly_sized' set←'set' training←'training'
- cousin `(dummy_cousin test/P/en)` → P1 — test←'testing'
- cousin `(cousin_he approach/C/en different/M/en fine_tuning/C/en training/C/en)` → P1 — approach←'approaches' different←'different' fine_tuning←'fine_tuning' training←'training'

Reading: 

## 18. PMC8545282.r1.L430.S1.U1

> 2) The deployed word embedding models in this work including BERT, GloVe and GloVe Twitter fail to capture the context.

- **parent** P1
  - `(dummy_sibling include/P/en)` — include←'including'
  - `(focal_he bert/C/focal glove/C/en twitter/C/en)` — bert←'BERT' glove←'GloVe GloVe' twitter←'Twitter'
- cousin `(dummy_cousin capture/P/en)` → P1 — capture←'capture'
- cousin `(dummy_cousin fail/P/en)` → P1 — fail←'fail'
- cousin `(cousin_he work/C/en)` → P1 — work←'work'
- cousin `(dummy_cousin deploy/P/en)` → P1 — deploy←'deployed'
- cousin `(cousin_he embedding/M/en model/C/en word/C/en)` → P1 — embedding←'embedding' model←'models' word←'word'
- cousin `(cousin_he context/C/en)` → P1 — context←'context'

Reading: 

## 19. PMC8502794.r1.L132.S2.U1

> Bidirectional encoder representations (BERT) is the latest NLP algorithm and is a part of the larger cloud machine learning API family from Google.

- **parent** P1
  - `(dummy_sibling be/P/en)` — be←'is'
  - `(focal_he bert/C/focal bidirectional/M/en encoder/C/en representation/C/en)` — bert←'BERT' bidirectional←'Bidirectional' encoder←'encoder' representation←'representations'
  - `(sibling_he algorithm/C/en late/M/en nlp/C/en)` — algorithm←'algorithm' late←'latest' nlp←'NLP'
- cousin `(dummy_cousin be/P/en)` → P1 — be←'is'
- cousin `(cousin_he api/C/en cloud/C/en family/C/en google/C/en large/M/en learning/C/en machine/C/en part/C/en)` → P1 — api←'API' cloud←'cloud' family←'family' google←'Google' large←'larger' learning←'learning' machine←'machine' part←'part'

Reading: 

## 20. PMC7721552.r1.L73.S6.U1

> Recently, deep learning models based on transformer architectures such as Bidirectional Encoder Representations from Transformers (BERT), XLNet, and Robustly optimized BERT approach (RoBERTa) have demonstrated state_of_the_art performances on the STS benchmark dataset and remarkably outperformed the previous models.

- **parent** P1
  - `(dummy_sibling base/P/en)` — base←'based'
  - `(focal_he approach/C/en architecture/C/en bert/C/focal optimize/P/en roberta/C/focal robustly/M/en transformer/C/en xlnet/C/en)` — approach←'approach' architecture←'architectures' bert←'BERT BERT Bidirectional Encoder Representations Transformers from' optimize←'optimized' roberta←'RoBERTa' robustly←'Robustly' transformer←'transformer' xlnet←'XLNet'
- **parent** P2
  - `(dummy_sibling optimize/P/en robustly/M/en)` — optimize←'optimized' robustly←'Robustly'
  - `(focal_he bert/C/focal)` — bert←'BERT BERT Bidirectional Encoder Representations Transformers from'
- cousin `(dummy_cousin demonstrate/P/en)` → P1, P2 — demonstrate←'demonstrated'
- cousin `(cousin_he deep/M/en learning/M/en model/C/en)` → P1, P2 — deep←'deep' learning←'learning' model←'models models'
- cousin `(cousin_he benchmark/C/en dataset/C/en performance/C/en state_of_the_art/C/en sts/C/en)` → P1, P2 — benchmark←'benchmark' dataset←'dataset' performance←'performances' state_of_the_art←'state_of_the_art' sts←'STS'
- cousin `(dummy_cousin outperform/P/en remarkably/M/en)` → P1, P2 — outperform←'outperformed' remarkably←'remarkably'
- cousin `(cousin_he model/C/en previous/M/en)` → P1, P2 — model←'models models' previous←'previous'
- cousin `(dummy_cousin base/P/en)` → P2 — base←'based'
- cousin `(cousin_he approach/C/en architecture/C/en transformer/C/en xlnet/C/en)` → P2 — approach←'approach' architecture←'architectures' transformer←'transformer' xlnet←'XLNet'

Reading: 

## 21. PMC8684061.r1.L74.S3.U1

> In their approach, they first employed Multilingual BERT as language representations, and then combined the character_level representation, part_of_speech (POS) representation and word shape representation of each word to the BERT representation.

- **parent** P1
  - `(dummy_sibling employ/P/en first/M/en)` — employ←'employed' first←'first'
  - `(focal_he bert/C/focal multilingual/C/en)` — bert←'BERT BERT' multilingual←'Multilingual'
  - `(sibling_he approach/C/en)` — approach←'approach'
  - `(sibling_he language/C/en representation/C/en)` — language←'language' representation←'representations'
- **parent** P2
  - `(dummy_sibling combine/P/en)` — combine←'combined'
  - `(focal_he bert/C/focal representation/C/en)` — bert←'BERT BERT' representation←'representation representation representation representation'
  - `(sibling_he character_level/C/en part_of_speech/C/en pos/C/en representation/C/en shape/C/en word/C/en)` — character_level←'character_level' part_of_speech←'part_of_speech' pos←'POS' representation←'representation representation representation representation' shape←'shape' word←'word word'
- cousin `(dummy_cousin combine/P/en)` → P1 — combine←'combined'
- cousin `(cousin_he character_level/C/en part_of_speech/C/en pos/C/en representation/C/en shape/C/en word/C/en)` → P1 — character_level←'character_level' part_of_speech←'part_of_speech' pos←'POS' representation←'representation representation representation representation' shape←'shape' word←'word word'
- cousin `(cousin_he representation/C/en)` → P1 — representation←'representation representation representation representation'
- cousin `(dummy_cousin employ/P/en first/M/en)` → P2 — employ←'employed' first←'first'
- cousin `(cousin_he approach/C/en)` → P2 — approach←'approach'
- cousin `(cousin_he multilingual/C/en)` → P2 — multilingual←'Multilingual'
- cousin `(cousin_he language/C/en representation/C/en)` → P2 — language←'language' representation←'representations'

Reading: 

## 22. PMC8049133.r1.L301.S1.U1

> Our research into the possibility of using feature_based classifiers for the detection of SOTA language models offers not only an understanding of the method’s general performance, but also delivers many insights into more general language model detection issues.

- **parent** P1
  - `(dummy_sibling use/P/en)` — use←'using'
  - `(focal_he detection/C/en language_model/C/focal sota/C/en)` — detection←'detection detection' language_model←'language language model models' sota←'SOTA'
  - `(sibling_he classifier/C/en feature_based/M/en)` — classifier←'classifiers' feature_based←'feature_based'
- **parent** P2
  - `(dummy_sibling deliver/P/en)` — deliver←'delivers'
  - `(focal_he detection/C/en general/M/en insight/C/en issue/C/en language_model/C/focal more/M/en)` — detection←'detection detection' general←'general general' insight←'insights' issue←'issues' language_model←'language language model models' more←'more'
- cousin `(dummy_cousin offer/P/en)` → P1, P2 — offer←'offers'
- cousin `(cousin_he our/M/en possibility/C/en research/C/en)` → P1, P2 — our←'Our' possibility←'possibility' research←'research'
- cousin `(cousin_he general/M/en method/C/en performance/C/en understanding/C/en)` → P1, P2 — general←'general general' method←'method' performance←'performance' understanding←'understanding'
- cousin `(dummy_cousin deliver/P/en)` → P1 — deliver←'delivers'
- cousin `(cousin_he detection/C/en general/M/en insight/C/en issue/C/en more/M/en)` → P1 — detection←'detection detection' general←'general general' insight←'insights' issue←'issues' more←'more'
- cousin `(dummy_cousin use/P/en)` → P2 — use←'using'
- cousin `(cousin_he classifier/C/en feature_based/M/en)` → P2 — classifier←'classifiers' feature_based←'feature_based'
- cousin `(cousin_he detection/C/en sota/C/en)` → P2 — detection←'detection detection' sota←'SOTA'

Reading: 

## 23. PMC8719412.r1.L252.S2.U1

> “FT-BERT” denotes fine_tuned BERT on special Chinese clinical corpus.

- **parent** P1
  - `(dummy_sibling denote/P/en)` — denote←'denotes'
  - `(focal_he bert/C/focal ft/C/en)` — bert←'BERT' ft←'FT'
  - `(sibling_he bert/C/en fine_tuned/M/en)` — bert←'BERT BERT' fine_tuned←'fine_tuned'
  - `(sibling_he chinese/M/en clinical/M/en corpus/C/en special/M/en)` — chinese←'Chinese' clinical←'clinical' corpus←'corpus' special←'special'

Reading: 

## 24. PMC7451563.r1.L127.S3.U1

> Second, the pre_trained BERT-based model addresses unintended bias in data or trained models and tries to mitigate the racial bias in our pre_trained BERT-based classifiers.

- **parent** P1
  - `(dummy_sibling address/P/en)` — address←'addresses'
  - `(focal_he based/M/en bert/C/focal model/C/en pre_trained/M/en)` — based←'based based' bert←'BERT BERT' model←'model' pre_trained←'pre_trained pre_trained'
  - `(sibling_he bias/C/en datum/C/en model/C/en train/P/en unintended/M/en)` — bias←'bias bias' datum←'data' model←'models' train←'trained' unintended←'unintended'
- **parent** P2
  - `(dummy_sibling mitigate/P/en try/P/en)` — mitigate←'mitigate' try←'tries'
  - `(focal_he based/M/en bert/C/focal bias/C/en classifier/C/en our/M/en pre_trained/M/en racial/M/en)` — based←'based based' bert←'BERT BERT' bias←'bias bias' classifier←'classifiers' our←'our' pre_trained←'pre_trained pre_trained' racial←'racial'
- cousin `(dummy_cousin try/P/en)` → P1 — try←'tries'
- cousin `(dummy_cousin mitigate/P/en)` → P1 — mitigate←'mitigate'
- cousin `(cousin_he based/M/en bias/C/en classifier/C/en our/M/en pre_trained/M/en racial/M/en)` → P1 — based←'based based' bias←'bias bias' classifier←'classifiers' our←'our' pre_trained←'pre_trained pre_trained' racial←'racial'
- cousin `(dummy_cousin address/P/en)` → P2 — address←'addresses'
- cousin `(cousin_he based/M/en model/C/en pre_trained/M/en)` → P2 — based←'based based' model←'model' pre_trained←'pre_trained pre_trained'
- cousin `(cousin_he bias/C/en datum/C/en unintended/M/en)` → P2 — bias←'bias bias' datum←'data' unintended←'unintended'
- cousin `(dummy_cousin train/P/en)` → P2 — train←'trained'
- cousin `(cousin_he model/C/en)` → P2 — model←'models'

Reading: 

## 25. PMC8497875.r1.L126.S2.U1

> We compared the performance of BERT, Bioclinical-BERT, PubMedBERT, RoBERTa, and DeBERTa models on relation extraction tasks based on the evaluation metric of n2c2 2018 Shared Task 2.

- **parent** P1
  - `(dummy_sibling compare/P/en)` — compare←'compared'
  - `(focal_he 2018/M/en bert/C/en bert/C/focal bioclinical/C/en deberta/C/en evaluation/C/en extraction/C/en metric/C/en model/C/en n2c2/C/en performance/C/en pubmedbert/C/focal relation/C/en roberta/C/focal shared/C/en task/C/en)` — 2018←'2018' bert←'BERT' bert←'BERT' bioclinical←'Bioclinical' deberta←'DeBERTa' evaluation←'evaluation' extraction←'extraction' metric←'metric' model←'models' n2c2←'n2c2' performance←'performance' pubmedbert←'PubMedBERT' relation←'relation' roberta←'RoBERTa' shared←'Shared' task←'Task tasks'

Reading: 

## 26. PMC7510341.r1.L45.S11.U1

> Thus the transformer model has become the most commonly used model in the seq2seq framework in recent years.

- **parent** P1
  - `(dummy_sibling become/P/en)` — become←'become'
  - `(focal_he transformer_model/C/focal)` — transformer_model←'model transformer'
  - `(sibling_he framework/C/en model/C/en most/M/en seq2seq/C/en use/P/en)` — framework←'framework' model←'model model' most←'most' seq2seq←'seq2seq' use←'used'
  - `(sibling_he recent/M/en year/C/en)` — recent←'recent' year←'years'

Reading: 

## 27. PMC7451563.r1.L264.S1.U1

> The x-axis is the portion of the training and validation sets used for training our BERT-based model and the baselines, the y-axis shows the F1-measure.

- **parent** P1
  - `(dummy_sibling train/P/en use/P/en)` — train←'training' use←'used'
  - `(focal_he based/M/en bert/C/focal model/C/en our/M/en)` — based←'based' bert←'BERT' model←'model' our←'our'
- cousin `(dummy_cousin show/P/en)` → P1 — show←'shows'
- cousin `(dummy_cousin be/P/en)` → P1 — be←'is'
- cousin `(cousin_he axis/C/en)` → P1 — axis←'axis axis'
- cousin `(cousin_he portion/C/en set/C/en training/C/en validation/C/en)` → P1 — portion←'portion' set←'sets' training←'training' validation←'validation'
- cousin `(cousin_he baseline/C/en)` → P1 — baseline←'baselines'
- cousin `(cousin_he f1/C/en measure/C/en)` → P1 — f1←'F1' measure←'measure'

Reading: 

## 28. PMC7831099.r1.L344.S4.U1

> In contrast, the BERT embedding was fine_tuned to keep the dropout probability at 0.1.

- **parent** P1
  - `(dummy_sibling be/M/en fine_tune/P/en)` — be←'was' fine_tune←'fine_tuned'
  - `(focal_he bert/C/focal embedding/C/en)` — bert←'BERT' embedding←'embedding'
  - `(sibling_he contrast/C/en)` — contrast←'contrast'
  - `(sibling_he 0%2e1/C/en dropout/C/en keep/P/en probability/C/en)` — 0%2e1←'0.1' dropout←'dropout' keep←'keep' probability←'probability'

Reading: 

## 29. PMC8600147.r1.L125.S4.U1

> Caswell et al. show the value of manual audits of multilingual corpora to highlight the dubious quality of many datasets used for language model training.

- **parent** P1
  - `(dummy_sibling use/P/en)` — use←'used'
  - `(focal_he language_model/C/focal training/C/en)` — language_model←'language model' training←'training'
- cousin `(dummy_cousin show/P/en)` → P1 — show←'show'
- cousin `(cousin_he caswell/C/en)` → P1 — caswell←'Caswell'
- cousin `(cousin_he audit/C/en corpus/C/en manual/M/en multilingual/M/en value/C/en)` → P1 — audit←'audits' corpus←'corpora' manual←'manual' multilingual←'multilingual' value←'value'
- cousin `(dummy_cousin highlight/P/en)` → P1 — highlight←'highlight'
- cousin `(cousin_he dataset/C/en dubious/M/en quality/C/en)` → P1 — dataset←'datasets' dubious←'dubious' quality←'quality'

Reading: 

## 30. PMC8259990.r1.L73.S2.U1

> In Multi_Conditional Language Model we describe the methodology behind the CBAG model, which specializes the transformer architecture for generating biomedical abstracts.

- **parent** P1
  - `(dummy_sibling describe/P/en)` — describe←'describe'
  - `(focal_he language_model/C/focal multi_conditional/C/en)` — language_model←'Language Model' multi_conditional←'Multi_Conditional'
  - `(sibling_he abstract/C/en architecture/C/en biomedical/M/en cbag/C/en generate/P/en methodology/C/en model/C/en specialize/P/en transformer/C/en)` — abstract←'abstracts' architecture←'architecture' biomedical←'biomedical' cbag←'CBAG' generate←'generating' methodology←'methodology' model←'model' specialize←'specializes' transformer←'transformer'

Reading: 
