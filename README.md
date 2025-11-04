# Explicit-Gradient-Linking-Mechanism-in-Neural-Networks-with-External-Memory
外部記憶を持つニューラルネットワークにおける明示的な勾配リンク機構

本リポジトリは、外部記憶型ニューラルネットワーク（NTM/DNCなど）における長年の根本的な課題、
すなわち、**「記憶への書き込みと読み出し操作の間の勾配フローが途絶える問題」**に対処するために提案された、
明示的な勾配リンク機構の実装コード、実験データ、論文を公開しています

1.提案手法の成果

私は時系列IDトークンを使用してこの途絶した勾配を再接続するメカニズムを提案し、実験により以下の顕著な成果を確認しました。

1.学習効率の劇的な向上: コピー課題において、従来のモデルが停滞する中で、
提案手法は圧倒的な速度と安定性で収束を達成しました。

2.協調的記憶戦略の自律的獲得: 勾配リンクが途絶を解消することで、
モデルは書き込みと読み出しが連携する効率的な記憶利用戦略を自律的に早く学習できるようになります。


2. 論文情報

論文タイトル: Explicit Gradient Linking Mechanism in Neural Networks with External Memory

著者: 増田 翔星 (Shousei Masuda)

関連分野: Deep Learning, External Memory, Neural Turing Machines (NTM), Differentiable Neural Computers (DNC), Gradient Flow Analysis

論文PDF: [[...](https://github.com/shouseimasuda1-cyber/Explicit-Gradient-Linking-Mechanism-in-Neural-Networks-with-External-Memory/blob/main/ExplicitGradientLinking/paper/Explicit_Gradient_Linking_Mechanism_in_Neural_Networks_with_External_Memory.pdf)]

arXiv投稿予定: 論文に問題がなく推薦者承認後に投稿予定


3. 実装
必要な環境 

Python 3.8+

PyTorch 1.10+

Matplotlib

実行方法

1.本リポジトリをクローンまたはダウンロードします。

2.以下のコマンドで学習を開始します。

>>>$python train_copy_task.py


3.学習の進行状況と、勾配リンクあり/なしの損失曲線の比較グラフが学習完了後、出力されます。

4. 注意
論文にも書かれていますが論文と全く同じデーターを再現することは難しいですが,
本研究手法を適応したモデルが学習スピードなどで一貫した優位に立ちます


Contact: shouseimasuda1@gmail.com
