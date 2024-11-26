package com.yarvin;

import org.apache.commons.math3.linear.*;

public class QrMethod {

    public static RealVector qr(RealMatrix A, RealVector b) {
        int m = A.getRowDimension();
        int n = A.getColumnDimension();

        // Создаем копии матриц R и Q
        RealMatrix R = A.copy();
        RealMatrix Q = MatrixUtils.createRealIdentityMatrix(m);

        for (int k = 0; k < n; k++) {
            // Извлекаем столбец x
            double[] xData = R.getSubMatrix(k, m - 1, k, k).getColumn(0);
            RealVector x = new ArrayRealVector(xData);

            // Вычисляем норму x
            double normX = x.getNorm();

            // Создаем вектор v
            RealVector v = x.copy();
            v.setEntry(0, v.getEntry(0) + Math.copySign(normX, x.getEntry(0)));
            v = v.mapDivide(v.getNorm());

            // Обновляем матрицу R
            RealMatrix vk = v.outerProduct(v).scalarMultiply(2.0);
            RealMatrix Rk = R.getSubMatrix(k, m - 1, k, n - 1);
            R.setSubMatrix(Rk.subtract(vk.multiply(Rk)).getData(), k, k);

            // Обновляем матрицу Q
            RealMatrix Qk = Q.getSubMatrix(0, m - 1, k, m - 1);
            Q.setSubMatrix(Qk.subtract(Qk.multiply(v.outerProduct(v).scalarMultiply(2.0))).getData(), 0, k);
        }

        // Вычисляем y = Q^T * b
        RealVector y = Q.transpose().operate(b);

        // Обратный ход для решения Rx = y
        double[] xSolution = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = y.getEntry(i);
            for (int j = i + 1; j < n; j++) {
                sum -= R.getEntry(i, j) * xSolution[j];
            }
            xSolution[i] = sum / R.getEntry(i, i);
        }

        return new ArrayRealVector(xSolution);
    }

    public static void main(String[] args) {
        // Определяем матрицу A
        double[][] AData = {
                {2.2, 4, -3, 1.5, 0.6, 2, 0.7},
                {4, 3.2, 1.5, -0.7, -0.8, 3, 1},
                {-3, 1.5, 1.8, 0.9, 3, 2, 2},
                {1.5, -0.7, 0.9, 2.2, 4, 3, 1},
                {0.6, -0.8, 3, 4, 3.2, 0.6, 0.7},
                {2, 3, 2, 3, 0.6, 2.2, 4},
                {0.7, 1, 2, 1, 0.7, 4, 3.2}
        };
        RealMatrix A = new Array2DRowRealMatrix(AData);

        // Определяем вектор b
        double[] bData = {3.2, 4.3, -0.1, 3.5, 5.3, 9.0, 3.7};
        RealVector b = new ArrayRealVector(bData);

        // Решаем систему уравнений с помощью QR-разложения
        RealVector x = qr(A, b);

        // Выводим решение
        System.out.print("Решение системы Ax = b: [");
        for (int i = 0; i < x.getDimension(); i++) {
            System.out.printf(" %.8f", x.getEntry(i));
        }
        System.out.println(" ]\n");

        // Вычисляем невязку решения
        RealVector residual = b.subtract(A.operate(x));
        double residualNorm = residual.getNorm();
        System.out.printf("Невязка решения (норма вектора b - Ax): %.11e%n", residualNorm);
    }
}
