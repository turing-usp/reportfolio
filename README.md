# Reportfólio do Grupo Turing, o grupo de IA da USP.
O Grupo Turing, grupo de IA da USP, tem como objetivo estudar, aplicar e difundir inteligência artifical no ecossistema de São Paulo, buscando se tornar uma referência em IA no Brasil. Neste repositório, apresentamos, na forma de awesome list, quem somos e nosso trabalho.

# Índice
  * [Quem somos](#Quem-somos)
  * [Projetos por área de estudo](#Projetos-por-área-de-estudo)
  * [Papers with code](#Papers-with-code)
  * [Visão Computacional](#Visão-Computacional)
  * [Aprendizado por reforço (Reinforcement Learning)](#aprendizado-por-reforço-reinforcement-learning)
  * [Processamento-de-Linguagem-Natural-(NLP)](#)
  * [Finanças quantitativas (Quant)](#Finanças-quantitativas-(Quant))
  * [Data Science](#Data-Science)
  
## Quem somos
Somos o grupo de extensão acadêmica da Universidade de São Paulo que estuda, dissemina e aplica conhecimentos de Inteligência Artificial.

### História
Surgimos em 2015 como um grupo de estudos originalmente idealizado por duas mulheres, fundado por um grupo de três politécnicos e batizado em homenagem a Alan Turing (1912-1954), matemático e lógico inglês considerado o pai da computação.

### Missão
Nossa missão é se tornar uma referência nacional em produção de conteúdo relacionado à Inteligência Artificial. Abrangendo material de estudo em português, a realização de eventos como workshops e até a aplicação em projetos.

### Cultura
Prezamos pela diversidade e o respeito ao próximo. Valorizamos a proatividade de nossos membros nos quesitos de entrega, liderança e a habilidade de trabalhar em equipe tomando atitudes, mas mantendo-se aberto a sugestões.

## Implementações limpas e explicadas de papers relevantes de IA:
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
 7. [Long Short-term memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
  * https://github.com/piEsposito/pytorch-lstm-by-hand
 8. [LSTM peephole connections](http://jmlr.org/papers/volume3/gers02a/gers02a.pdf)
  * Também tomamos como referência: [Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf)
  * https://github.com/piEsposito/blitz-bayesian-deep-learning/blob/master/blitz/modules/lstm_bayesian_layer.py - como uma opção para a operação feedforward
 9. [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555)
  * https://github.com/piEsposito/blitz-bayesian-deep-learning/blob/master/blitz/modules/gru_bayesian_layer.py
 
## Visão Computacional

### O que é:

Como humanos somos capazes de enteder conteudos de imagens facilmente, mas e o computador? Como ele enxerga uma imagem? Como ele consegue discernir o seu conteudo e tirar informações dela?
Visão Computacional é a aréa de estudos focada em estudar justamente isso e como aproveitar todo o potencial de um computador tem de analisar imagens em um tempo infinitamente menor que pessoas.
Usamos todo esse potencial para criar aplicações que melhorem processos antes não automatizados.
Atualmente, trabalhamos em um sistema de reconhecimento e codificação de rostos para uma empresa parceira.

### O que já fizemos:

- Projeto que visa converter sinais da Língua Brasileira de Sinais em Português: *https://bitbucket.org/grupoturing/libras_codes/src/master/
- Jo-Ken-Po: contar o número de dedos apresenta em frente da webcam: 
*https://bitbucket.org/grupoturing/contador-de-dedos/src/master/
- Workshop de Classificação de Imagens.
- Posts do Medium sobre CV
- Classificação com redes neurais: 
*https://medium.com/turing-talks/turing-talks-22-modelos-de-predi%C3%A7%C3%A3o-redes-neurais-parte-3-9c5d5d0c60e7
- Redes neurais convolucionais: 
*https://medium.com/turing-talks/turing-talks-23-modelos-de-predi%C3%A7%C3%A3o-redes-neurais-convolucionais-d364654a34de

O que estamos fazendo:
- Estudos de técnicas de reconhecimento de imagens, como as grandes empresas implementam algoritmos de classificação etentamos nos atualizar nas técnicas atuais de CV.
- Projeto de reconhecimento facial com codificação não reversível do rosto humano, visando a recente LGPD.

## Aprendizado por reforço (Reinforcement Learning)

### O que é:
Em poucas palavras, Reinforecemnt Learning é uma modalidade de Machine Learning em que o modelo matemático, enquanto parte decisória de um agente, 'aprende fazendo': iniciado aleatoriamente, o modelo aprende tomando ações e escolhendo a que tem melhor resultado. Depois do treinamento, o modelo passa a encontrar a ação com melhor resultado e o agente passa a ser apto a realizar a tarefa que está fazendo. 

### O que já fizemos:

 1. Workshop Introdutório de Aprendizado por Reforço:
  * [Material](https://github.com/GrupoTuring/Workshop-de-Aprendizado-por-Reforco)
  * [Vídeo Completo](https://youtu.be/FxcWqI-l29E)
 2. Rede neural que aprende a jogar Super Mario Bros:
  * [Código](https://github.com/Berbardo/MarioRL)
  * [Post](https://medium.com/turing-talks/usando-deep-learning-para-jogar-super-mario-bros-8d58eee6e9c2)
 3. Agente que aprende a jogar Pong com Q-Learning:
  * [Código](https://github.com/GrupoTuring/Turing-Talks/tree/master/Aprendizado%20por%20Refor%C3%A7o/QLearningTabular)
  * [Post](https://medium.com/turing-talks/criando-uma-ia-que-aprende-a-jogar-pong-f379b0170017)
 4. Seríe de posts introdutórios sobre o tema em nosso Medium:
  * [Introdução](https://medium.com/turing-talks/aprendizado-por-refor%C3%A7o-1-introdu%C3%A7%C3%A3o-7382ebb641ab)
  * [Processo de Decisão de Markov: Parte 1](https://medium.com/turing-talks/aprendizado-por-refor%C3%A7o-2-processo-de-decis%C3%A3o-de-markov-mdp-parte-1-84e69e05f007)
  * [Processo de Decisão de Markov: Parte 2](https://medium.com/turing-talks/aprendizado-por-refor%C3%A7o-3-processo-de-decis%C3%A3o-de-markov-parte-2-15fe4e2a4950)
  * [Gym](https://medium.com/turing-talks/aprendizado-por-refor%C3%A7o-4-gym-d18ac1280628)
  * [Programação Dinâmica](https://medium.com/turing-talks/aprendizado-por-refor%C3%A7o-5-programa%C3%A7%C3%A3o-din%C3%A2mica-8db4db386b67)
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
### O que é:
A área de Data Science, nova no grupo, tem como objetivo aplicar o conhecimento desenvolvido dentro do grupo para resolver problemas reais dentro de organizações. Alinhando-se com a filosofia do grupo, a área procura sempre se aproximar de ONGs, com o intuito de produzir resultados que tenham um impacto positivo na sociedade.
### O que fazemos:
No momento, temos trabalhos em andamento com entidades públicas e privadas que serão publicados e mostrados aqui quanto estiverem prontos.
