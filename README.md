# Reportfólio do Grupo Turing, o grupo de IA da USP.
O Grupo Turing, grupo de IA da USP, tem como objetivo estudar, aplicar e difundir inteligência artifical no ecossistema de São Paulo, buscando se tornar uma referência em IA no Brasil. Neste repositório, apresentamos, na forma de awesome list, quem somos e nosso trabalho.

# Índice
  * [Quem somos](#Quem-somos)
  * [Projetos por área de estudo](#Projetos-por-área-de-estudo)
  * [Papers with code](#Papers-with-code)
  * [Visão Computacional](#Visão-Computacional)
  * [Aprendizado por reforço (Reinforcement Learning)](#Aprendizado-por-reforço-(Reinforcement-Learning))
  * [Processamento-de-Linguagem-Natural-(NLP)](#)
  * [Finanças quantitativas (Quant)](#Finanças-quantitativas-(Quant))
  * [Data Science](#Data-Science)
  
## Quem somos

## Projetos por área de estudo

## Papers with code
### O que é:
A motivação deste reportfólio foi criar um lugar onde houvesse implementações de referência de IA feito por falantes de língua portuguesa para falantes de língua portuguesa - e para isso escolheríamos papers relevantes de Deep Learning e Machine Learning e os implementaríamos num repositório do GitHub.

Nesta seção do reportfólio, membros do grupo escolhem artigos ou ideias interessantes em Deep Learning e as implementam de forma simples e explicada, eventualmente extentendo a ideia. No final, nem tudo ficou em portugês... Acontece...
### O que implementamos:
 1. [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424) - se tornou uma lib em PyTorch de Deep Learning com inferência variacional, BLiTZ.
  * https://towardsdatascience.com/bayesian-lstm-on-pytorch-with-blitz-a-pytorch-bayesian-deep-learning-library-5e1fec432ad3
  * https://github.com/piEsposito/blitz-bayesian-deep-learning
 2. [Curvilinear Component Analysis implementation for Python](https://ieeexplore.ieee.org/document/554199) - um método para redução de dimensionalidade não linear.
  * https://github.com/FelipeAugustoMachado/Curvilinear-Component-Analysis-Python
 3. Auto-Encoder no PyTorch
  * https://medium.com/turing-talks/redes-neurais-autoencoders-com-pytorch-fbce7338e5de
  * https://github.com/paulosestini/AutoEncoder
 4. [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) - O trabalho que inaugurou a área de Deep Reinforcement Learning, tão hypada (com razão) nos dias de hoje.
  * https://github.com/lucms/DQN
 5.  [Policy Gradients for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)- Implementação de REINFORCE - Policy Gradients no VizDoom:
  * https://medium.com/@piero.skywalker/reinforcement-learning-para-al%C3%A9m-do-cartpole-policy-gradients-no-vizdoom-720d81ee3cb5
  * https://github.com/piEsposito/policy-gradients-doom
 6. [Bayesian Recurrent Neural Networks](https://arxiv.org/pdf/1704.02798.pdf) - Redes Neurais Recorrentes como parte da lib de Deep Learning Bayesiano
  * https://towardsdatascience.com/bayesian-lstm-on-pytorch-with-blitz-a-pytorch-bayesian-deep-learning-library-5e1fec432ad3
  * https://github.com/piEsposito/blitz-bayesian-deep-learning/blob/master/blitz/modules/lstm_bayesian_layer.py
 
## Visão Computacional

## Aprendizado por reforço (Reinforcement Learning)

### O que é:
Em poucas palavras, Reinforecemnt Learning é uma modalidade de Machine Learning em que o modelo matemático, enquanto parte decisória de um agente, 'aprende fazendo': iniciado aleatoriamente, o modelo aprende tomando ações e escolhendo a que tem melhor resultado. Depois do treinamento, o modelo passa a encontrar a ação com melhor resultado e o agente passa a ser apto a realizar a tarefa que está fazendo. 

### O que já fizemos:

 1. Seríe de posts introdutórios sobre o tema em nosso Medium:
  * https://medium.com/turing-talks/aprendizado-por-refor%C3%A7o-1-introdu%C3%A7%C3%A3o-7382ebb641ab
  * https://medium.com/turing-talks/aprendizado-por-refor%C3%A7o-2-processo-de-decis%C3%A3o-de-markov-mdp-parte-1-84e69e05f007
  * https://medium.com/turing-talks/aprendizado-por-refor%C3%A7o-3-processo-de-decis%C3%A3o-de-markov-parte-2-15fe4e2a4950
  * https://medium.com/turing-talks/aprendizado-por-refor%C3%A7o-4-gym-d18ac1280628
  * https://medium.com/turing-talks/aprendizado-por-refor%C3%A7o-5-programa%C3%A7%C3%A3o-din%C3%A2mica-8db4db386b67
### O que estamos fazendo:
 * Estudo dos métodos e técnicas de RL, como DQN, PG, AC, ... e como modelar os problemas, implementando esses métodos em ambientes do gym e outros jogos. 

 * Projeto: controle de um carrinho usando RL, pra fazer ele seguir um caminho ou desviar de obstáculos (estamos num estado inicial ainda), envolvendo a simulação do agente e criação de modelo físico (após a quarentena)


## Processamento de Linguagem Natural (NLP)
### O que é:
Processamento de Linguagem Natural (do inglês Natural Language Processing - NLP) é uma área da Inteligência Artificial que estuda a intepretação e manipulação de linguagens humanas - naturais por computadores. Entre suas aplicações estão:
 * Interpretação de texto;
 * Criação de conteúdo, e até
 * Tradução de texto

### O que já fizemos:
 1. Uso de Machine Learning para classificação dos heterônimos de Fernando Pessoa
  * https://medium.com/turing-talks/como-machine-learning-consegue-diferenciar-heter%C3%B4nimos-de-fernando-pessoa-156d0d52a478
  * https://github.com/GrupoTuring/fernando-pessoa
  
 2. Análise de sentimento em texto com Deep Learning
  * https://medium.com/turing-talks/an%C3%A1lise-de-sentimento-usando-lstm-no-pytorch-d90f001eb9d7
  * https://github.com/piEsposito/nlp-sentiment-analysis-turing-talks
 
 3. Post no Medium de introdução à NLP
  * https://medium.com/turing-talks/introdu%C3%A7%C3%A3o-ao-processamento-de-linguagem-natural-com-baco-exu-do-blues-17cbb7404258
 
### O que estamos fazendo: 
Temos projetos em andamento relacionados a interpretação de textos curtos para extração de insights.

## Finanças quantitativas (Quant)

### O que é:
Aplicações de programação e inteligência artificial no mercado financeiro, estudando finanças, aplicações e métodos quantitativos. Tudo isso através de cursos, implementações de papers, competições, e visitas a gestoras quantitativas e outras empresas do setor.
### O que já fizemos:
 1. Repositório no GitHub com script utilitário para obtenção de dados do mercado financeiro:
  * https://github.com/GrupoTuring/Quant-Utils
 2. Lib em Python com estratégias de investimento do Turing Quant:
  * https://github.com/GrupoTuring/Quant-Models
  
## Data Science
