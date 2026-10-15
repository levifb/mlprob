# mlprob

Modelos de Aprendizagem de Máquina Probabilística implementados do zero.

Aqui ficam as listas da disciplina de ML probabilístico, feitas em Python (numpy, matplotlib, scipy e um pouco de sklearn só pra montar os dados). As fórmulas seguem a notação dos slides do professor.

## Listas

**Lista 1: modelos discretos**
- `questao1.ipynb`: Beta-Bernoulli. Estimativas ML, MAP e média da posteriori pra uma moeda (8 caras em 10), com gráficos de priori, verossimilhança e posteriori para Beta(1,1) e Beta(2,2).
- `questao2.ipynb`: Naive Bayes de Bernoulli bayesiano em cima de reviews da Amazon (`cleaned_reviews.csv`), com 20 repetições de treino/teste e prioris Beta(0.5,0.5), Beta(1,1) e Beta(2,2).

**Lista 2: regressão bayesiana**
- `questao1.ipynb`: regressão linear.
- `questao2.ipynb`: regressão polinomial de grau 5.
- `questao3.ipynb`: regressão logística, com solução MAP por IRLS.

**Lista 3: clustering**
- `questao1.ipynb`: mistura de gaussianas com solução MAP, treinada por EM.

## Material didático

Slides das aulas usados como referência (mixture models, regressão linear e regressão logística). Os enunciados de cada lista também estão nas pastas.

## Como rodar

Precisa de Python 3 com `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn` e `jupyter`. Abre o notebook dentro da pasta da lista, porque os CSVs são lidos com caminho relativo.

```
pip install numpy pandas matplotlib scipy scikit-learn jupyter
jupyter notebook
```

Autor: Levi Félix Barbosa
