package com.yarvin;

import java.text.DecimalFormat;

public class Lab4VinnitskiyYaroslav {

    public static void main(String[] args) {
        // Определение матрицы A и вектора b для решения системы A * x = b
        double[][] A = {
                {0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245},
                {0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101},
                {0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321},
                {0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183},
                {0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423},
                {0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923},
                {0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105},
        };
        double[] b = {0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673};

        int n = A.length;

        double[][] L = new double[n][n]; // Инициализируем нижнюю треугольную матрицу L
        double[][] U = new double[n][n]; // Инициализируем верхнюю треугольную матрицу U

        // Разложение матрицы A на L и U
        luDecomposition(A, L, U);

        // Решение системы L * y = b методом прямой подстановки
        double[] y = forwardSubstitution(L, b);
        // Решение системы U * x = y методом обратной подстановки
        double[] x = backwardSubstitution(U, y);

        System.out.println("Решение системы x:");
        printArray(x); // Вывод решения системы

        // Проверка правильности решения, вычисляя произведение A * x
        double[] Ax = multiplyMatrixVector(A, x);
        System.out.println("\nПроверка (A * x):");
        printArray(Ax); // Вывод A * x
        System.out.println("\nВектор свободных членов b:");
        printArray(b); // Вывод исходного вектора b
    }

    // Метод для разложения матрицы A на L и U
    public static void luDecomposition(double[][] A, double[][] L, double[][] U) {
        int n = A.length;

        for (int i = 0; i < n; i++) {
            // Вычисление элементов верхней треугольной матрицы U
            for (int k = i; k < n; k++) {
                double sumUpper = 0;
                for (int j = 0; j < i; j++) {
                    sumUpper += L[i][j] * U[j][k];
                }
                U[i][k] = A[i][k] - sumUpper; // U[i][k] - элемент верхней треугольной матрицы
            }

            // Установка диагонального элемента матрицы L
            L[i][i] = 1;

            // Вычисление элементов нижней треугольной матрицы L
            for (int k = i + 1; k < n; k++) {
                double sumLower = 0;
                for (int j = 0; j < i; j++) {
                    sumLower += L[k][j] * U[j][i];
                }
                L[k][i] = (A[k][i] - sumLower) / U[i][i]; // L[k][i] - элемент нижней треугольной матрицы
            }
        }
    }

    // Метод для прямой подстановки (решение L * y = b)
    public static double[] forwardSubstitution(double[][] L, double[] b) {
        int n = b.length;
        double[] y = new double[n]; // Инициализация вектора y

        // Вычисление каждого элемента y[i]
        for (int i = 0; i < n; i++) {
            double sumForward = 0;
            for (int j = 0; j < i; j++) {
                sumForward += L[i][j] * y[j];
            }
            y[i] = b[i] - sumForward; // Нахождение y[i]
        }

        return y; // Возвращаем вектор y
    }

    // Метод для обратной подстановки (решение U * x = y)
    public static double[] backwardSubstitution(double[][] U, double[] y) {
        int n = y.length;
        double[] x = new double[n]; // Инициализация вектора x

        // Вычисление каждого элемента x[i]
        for (int i = n - 1; i >= 0; i--) {
            double sumBackward = 0;
            for (int j = i + 1; j < n; j++) {
                sumBackward += U[i][j] * x[j];
            }
            x[i] = (y[i] - sumBackward) / U[i][i]; // Нахождение x[i]
        }

        return x; // Возвращаем вектор x
    }

    // Умножение матрицы на вектор
    public static double[] multiplyMatrixVector(double[][] A, double[] x) {
        int n = x.length;
        double[] result = new double[n];

        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                sum += A[i][j] * x[j];
            }
            result[i] = sum; // Элемент результата
        }

        return result; // Возвращаем результат умножения
    }

    // Вывод вектора с форматированием
    public static void printArray(double[] array) {
        DecimalFormat df = new DecimalFormat("#.###"); // Формат до 3 знаков после запятой
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < array.length; i++) {
            sb.append(df.format(array[i]));
            if (i < array.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        System.out.println(sb.toString()); // Вывод результата
    }
}
