package com.yarvin;

import org.apache.commons.math3.linear.*;

public class OptimalElimination {

    // Метод для решения системы линейных уравнений методом оптимального исключения
    public static double[] optimalExclusion(double[][] a, double[] b) {
        int n = a.length;
        double[][] augmentedMatrix = new double[n][n + 1];

        // Формируем расширенную матрицу (добавляем столбец правой части b)
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, augmentedMatrix[i], 0, n);
            augmentedMatrix[i][n] = b[i];
        }

        int[] selectedCols = new int[n]; // Массив для хранения индексов выбранных столбцов
        for (int i = 0; i < n; i++) {
            double max = 0;
            int colIndex = -1;

            // Поиск столбца с максимальным значением по модулю для текущей строки
            for (int j = 0; j < n; j++) {
                boolean isSelected = false;
                for (int k = 0; k < i; k++) {
                    if (selectedCols[k] == j) {
                        isSelected = true;
                        break;
                    }
                }
                if (!isSelected && Math.abs(augmentedMatrix[i][j]) > max) {
                    max = Math.abs(augmentedMatrix[i][j]);
                    colIndex = j;
                }
            }

            // Если столбец не найден, матрица вырождена
            if (colIndex == -1) {
                throw new SingularMatrixException();
            }

            selectedCols[i] = colIndex; // Сохраняем индекс выбранного столбца

            double pivot = augmentedMatrix[i][colIndex]; // Опорный элемент

            // Делим строку на опорный элемент
            for (int j = 0; j <= n; j++) {
                augmentedMatrix[i][j] /= pivot;
            }

            // Прямой ход метода Гаусса: зануляем остальные элементы столбца
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmentedMatrix[k][colIndex];
                    for (int j = 0; j <= n; j++) {
                        augmentedMatrix[k][j] -= factor * augmentedMatrix[i][j];
                    }
                }
            }
        }

        // Формируем вектор решения
        double[] solution = new double[n];
        for (int i = 0; i < n; i++) {
            solution[selectedCols[i]] = augmentedMatrix[i][n];
        }

        return solution;
    }

    public static void main(String[] args) {
        // Тестовая матрица: диагональная
        double[][] diagonalMatrix = {
                {1, 0, 0, 0, 0, 0, 0, 0},
                {0, 5, 0, 0, 0, 0, 0, 0},
                {0, 0, 3, 0, 0, 0, 0, 0},
                {0, 0, 0, 3, 0, 0, 0, 0},
                {0, 0, 0, 0, 7, 0, 0, 0},
                {0, 0, 0, 0, 0, 6, 0, 0},
                {0, 0, 0, 0, 0, 0, 8, 0},
                {0, 0, 0, 0, 0, 0, 0, 4}
        };
        double[] diagonalVector = {6, 6, 3, 3, 3, 9, 4, 5};

        // Тестовая матрица: симметричная
        double[][] symmetricMatrix = {
                {9, 5, 6, 6, 3, 8, 1, 2},
                {5, 9, 8, 3, 7, 6, 3, 7},
                {6, 8, 7, 8, 5, 7, 2, 4},
                {6, 3, 8, 7, 3, 4, 2, 8},
                {3, 7, 5, 3, 3, 5, 6, 6},
                {8, 6, 7, 4, 5, 9, 4, 8},
                {1, 3, 2, 2, 6, 4, 4, 7},
                {2, 7, 4, 8, 6, 8, 7, 7}
        };
        double[] symmetricVector = {1, 1, 8, 6, 6, 5, 2, 5};

        // Тестовая матрица: единичная
        double[][] identityMatrix = new double[8][8];
        for (int i = 0; i < 8; i++) {
            identityMatrix[i][i] = 1.0;
        }
        double[] identityVector = {9, 8, 9, 3, 1, 4, 5, 7};

        // Тестовая матрица: верхнетреугольная
        double[][] upperTriangularMatrix = {
                {6, 1, 8, 5, 1, 3, 7, 4},
                {0, 1, 1, 4, 7, 7, 1, 7},
                {0, 0, 9, 2, 5, 5, 4, 7},
                {0, 0, 0, 4, 5, 2, 3, 3},
                {0, 0, 0, 0, 6, 1, 8, 2},
                {0, 0, 0, 0, 0, 8, 7, 6},
                {0, 0, 0, 0, 0, 0, 5, 1},
                {0, 0, 0, 0, 0, 0, 0, 3}
        };
        double[] upperTriangularVector = {2, 4, 2, 7, 6, 1, 3, 3};

        // Тестовая матрица: произвольная
        double[][] randomMatrix = {
                {7, 5, 3, 7, 2, 9, 2, 5},
                {2, 5, 7, 9, 1, 5, 3, 5},
                {2, 3, 7, 4, 9, 9, 7, 8},
                {7, 9, 5, 5, 6, 4, 8, 8},
                {1, 9, 6, 4, 7, 1, 5, 8},
                {6, 8, 8, 8, 6, 6, 5, 2},
                {4, 6, 9, 3, 7, 8, 4, 7},
                {9, 4, 5, 7, 6, 7, 3, 8}
        };
        double[] randomVector = {9, 5, 8, 5, 7, 6, 4, 8};

        var matrices = new double[][][]{diagonalMatrix, symmetricMatrix, identityMatrix, upperTriangularMatrix, randomMatrix};
        var vectors = new double[][]{diagonalVector, symmetricVector, identityVector, upperTriangularVector, randomVector};

        for (int i = 0; i < matrices.length; i++) {
            System.out.printf("Матрица A_%d и вектор f_%d:\n", i + 1, i + 1);

            double[][] A = matrices[i];
            double[] b = vectors[i];

            // Решение системы методом главного элемента
            double[] x = optimalExclusion(A, b);

            // Вывод решения
            System.out.print("Решение: [");
            for (int j = 0; j < x.length; j++) {
                System.out.printf(" %.8f", x[j]);
                if (j < x.length - 1) {
                    System.out.print(",");
                }
            }
            System.out.println(" ]");

            // Проверка с помощью Apache Commons Math
            RealMatrix matrixA = new Array2DRowRealMatrix(A);
            RealVector vectorB = new ArrayRealVector(b);
            DecompositionSolver solver = new LUDecomposition(matrixA).getSolver();
            RealVector result = solver.solve(vectorB);

            System.out.print("Решение LuDecomposition: [");
            for (int j = 0; j < result.getDimension(); j++) {
                System.out.printf(" %.8f", result.getEntry(j));
                if (j < result.getDimension() - 1) {
                    System.out.print(",");
                }
            }
            System.out.println(" ]");

            // Вычисление норм
            RealVector realVectorX = new ArrayRealVector(x);
            double normDiff = result.subtract(realVectorX).getNorm();
            double normAxMinusB = matrixA.operate(result).subtract(vectorB).getNorm();

            // Вывод норм
            System.out.printf("Норма разности между решениями: %.15e\n", normDiff);
            System.out.printf("Норма разности (Ax - b): %.15e\n", normAxMinusB);

            System.out.println();
        }
    }
}