package com.yarvin;

import java.util.Arrays;

public class GaussElimination {

    public static double[] solve(double[][] A, double[] b) {
        int n = b.length;

        // Создаем расширенную матрицу
        double[][] M = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, M[i], 0, n);
            M[i][n] = b[i];
        }

        // Прямой ход
        for (int k = 0; k < n; k++) {
            // Выбор главного элемента по столбцу
            int maxIndex = k;
            for (int i = k + 1; i < n; i++) {
                if (Math.abs(M[i][k]) > Math.abs(M[maxIndex][k])) {
                    maxIndex = i;
                }
            }

            // Проверка на вырожденность матрицы
            if (Math.abs(M[maxIndex][k]) < 1e-10) {
                throw new IllegalArgumentException("Матрица вырождена.");
            }

            // Обмен строк, если требуется
            if (maxIndex != k) {
                double[] temp = M[k];
                M[k] = M[maxIndex];
                M[maxIndex] = temp;
            }

            // Прямой ход метода Гаусса
            for (int i = k + 1; i < n; i++) {
                double factor = M[i][k] / M[k][k];
                for (int j = k; j <= n; j++) {
                    M[i][j] -= factor * M[k][j];
                }
            }
        }

        // Обратный ход для нахождения решения
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0;
            for (int j = i + 1; j < n; j++) {
                sum += M[i][j] * x[j];
            }
            x[i] = (M[i][n] - sum) / M[i][i];
        }

        return x;
    }

    public static double calculateNorm(double[] vector) {
        double norm = 0.0;
        for (double v : vector) {
            norm += v * v;
        }
        return Math.sqrt(norm);
    }

    public static void main(String[] args) {
        // Диагональная матрица и вектор решений
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

        // Симметричная матрица и вектор решений
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

        // Единичная матрица и вектор решений
        double[][] identityMatrix = new double[8][8];
        for (int i = 0; i < 8; i++) {
            identityMatrix[i][i] = 1.0;
        }
        double[] identityVector = {9, 8, 9, 3, 1, 4, 5, 7};

        // Верхнетреугольная матрица и вектор решений
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

        // Случайная матрица и вектор решений
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

        // Решение для диагональной матрицы
        double[] diagonalSolution = solve(diagonalMatrix, diagonalVector);
        System.out.println("Решение для диагональной матрицы:");
        System.out.println(Arrays.toString(diagonalSolution));
        System.out.println("Норма: " + calculateNorm(diagonalSolution));

        // Решение для симметричной матрицы
        double[] symmetricSolution = solve(symmetricMatrix, symmetricVector);
        System.out.println("\nРешение для симметричной матрицы:");
        System.out.println(Arrays.toString(symmetricSolution));
        System.out.println("Норма: " + calculateNorm(symmetricSolution));

        // Решение для единичной матрицы
        double[] identitySolution = solve(identityMatrix, identityVector);
        System.out.println("\nРешение для единичной матрицы:");
        System.out.println(Arrays.toString(identitySolution));
        System.out.println("Норма: " + calculateNorm(identitySolution));

        // Решение для верхнетреугольной матрицы
        double[] upperTriangularSolution = solve(upperTriangularMatrix, upperTriangularVector);
        System.out.println("\nРешение для верхнетреугольной матрицы:");
        System.out.println(Arrays.toString(upperTriangularSolution));
        System.out.println("Норма: " + calculateNorm(upperTriangularSolution));

        // Решение для случайной матрицы
        double[] randomSolution = solve(randomMatrix, randomVector);
        System.out.println("\nРешение для случайной матрицы:");
        System.out.println(Arrays.toString(randomSolution));
        System.out.println("Норма: " + calculateNorm(randomSolution));
    }
}
