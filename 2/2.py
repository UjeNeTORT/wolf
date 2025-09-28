# %%
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import matplotlib.pyplot as plt

# %%
Array2D = NDArray[np.floating]
Array1D = np.ndarray[Tuple[int], np.dtype[np.floating]]

# %% [markdown]
# $$||X|| = \max\limits_{1 \leq j \leq n} \Sigma_{i} |X_{ij}|$$

# %%
def normMat(X: Array2D) -> np.floating:
  return np.max([[np.sum([np.abs(X[i][j]) for i in range(len(X[j]))])] for j in range(len(X[0]))])

def normVec(x: Array1D) -> np.floating:
  return np.sum([np.abs(x[j]) for j in range(len(x))])

# %%
def getG(A:  NDArray[np.floating],
         x0: NDArray[np.floating],
         b:  NDArray[np.floating]) -> NDArray[np.floating]:
  g = A @ x0 - b
  assert(len(A) == len(x0))
  # for i in range(len(x0)):
  return g

# %%
def inputMatrix(text: str) -> NDArray[np.floating]:
  n, m = map(int, input("Введите число строк и столбцов:").split()) # taking number of rows and column
  print(text)
  array = np.array([input().strip().split() for _ in range(n)], float)
  return array

def inputVector(text: str) -> NDArray[np.floating]:
  print(text)
  array = np.array(input().strip().split(), float)
  return array

# %%
A = inputMatrix("A: ")
assert(A.all)
# print(A)

b = inputVector("b: ")
assert(b.all)
# print(b)

x0 = inputVector("x0:")
if (x0.size == 0): x0 = np.zeros(len(b))
# print(x0)

x_ref = inputVector("x_ref: ")
if (x_ref.size == 0): x_ref = np.zeros(len(b))
# print(x_ref)

tolerance = float(input("Tolerance: ") or 1e-4)
# print(tolerance)


# %%
MaxItNum = 100

# %%
x = x0
g = getG(A, x, b)
i = 0

graph_data = list()
while(normVec(g) > tolerance and i < MaxItNum):
  mu = (g.T @ g) / (g.T @ A @ g)
  x = x - mu * g
  g = getG(A, x, b)
  i += 1
  graph_data.append(np.abs(x - x_ref))
  if (True): print(str(i) + ":", x)


# %%
plt.xlabel("", fontsize=12)
plt.ylabel(r"", fontsize=12)

plt.grid(True, which="both", ls="--", linewidth=0.7, alpha=0.7)
plt.scatter(range(len(graph_data)), [normVec(d) for d in graph_data])

plt.title(r"", fontsize=14)
plt.show()


