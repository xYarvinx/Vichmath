package com.yarvin;

import org.apache.commons.math3.linear.*;

public class BorderingMethod {

    // Метод для нахождения обратной матрицы с использованием метода окаймления
    public static RealMatrix getInv(RealMatrix A) {
        int n = A.getRowDimension();
        RealMatrix A_inv = new Array2DRowRealMatrix(n, n);  // Инициализация матрицы для обратной

        // Итерация по каждому размеру матрицы от 1 до n
        for (int i = 0; i < n; i++) {
            RealMatrix A_i = A.getSubMatrix(0, i, 0, i);  // Создаем подматрицу размера i+1
            if (i == 0) {
                // Для 1x1 матрицы инвертируем элемент
                A_inv.setEntry(0, 0, 1.0 / A_i.getEntry(0, 0));
            } else {
                // Разделение матрицы на блоки для окаймления
                RealMatrix A11 = A_inv.getSubMatrix(0, i - 1, 0, i - 1);
                RealMatrix a12 = A.getSubMatrix(0, i - 1, i, i);
                RealMatrix a21 = A.getSubMatrix(i, i, 0, i - 1);
                double alpha = A.getEntry(i, i);

                // Вычисляем величину S, необходимую для нахождения блоков обратной матрицы
                double S = alpha - a21.multiply(A11).multiply(a12).getEntry(0, 0);

                // Обновляем блоки обратной матрицы
                RealMatrix B11_new = A11.add(A11.multiply(a12).multiply(a21).multiply(A11).scalarMultiply(1.0 / S));
                RealMatrix b12_new = A11.multiply(a12).scalarMultiply(-1.0 / S);
                RealMatrix b21_new = a21.multiply(A11).scalarMultiply(-1.0 / S);
                RealMatrix beta_new = new Array2DRowRealMatrix(new double[][]{{1.0 / S}});

                // Сборка новой обратной матрицы из блоков
                RealMatrix top = MatrixUtils.createRealMatrix(B11_new.getRowDimension(), B11_new.getColumnDimension() + b12_new.getColumnDimension());
                for (int r = 0; r < B11_new.getRowDimension(); r++) {
                    for (int c = 0; c < B11_new.getColumnDimension(); c++) {
                        top.setEntry(r, c, B11_new.getEntry(r, c));
                    }
                    for (int c = 0; c < b12_new.getColumnDimension(); c++) {
                        top.setEntry(r, B11_new.getColumnDimension() + c, b12_new.getEntry(r, c));
                    }
                }

                RealMatrix bottom = MatrixUtils.createRealMatrix(b21_new.getRowDimension(), B11_new.getColumnDimension() + beta_new.getColumnDimension());
                for (int r = 0; r < b21_new.getRowDimension(); r++) {
                    for (int c = 0; c < b21_new.getColumnDimension(); c++) {
                        bottom.setEntry(r, c, b21_new.getEntry(r, c));
                    }
                    for (int c = 0; c < beta_new.getColumnDimension(); c++) {
                        bottom.setEntry(r, B11_new.getColumnDimension() + c, beta_new.getEntry(r, c));
                    }
                }

                // Объединение top и bottom для создания новой версии A_inv
                A_inv = MatrixUtils.createRealMatrix(top.getRowDimension() + bottom.getRowDimension(), top.getColumnDimension());
                for (int r = 0; r < top.getRowDimension(); r++) {
                    for (int c = 0; c < top.getColumnDimension(); c++) {
                        A_inv.setEntry(r, c, top.getEntry(r, c));
                    }
                }
                for (int r = 0; r < bottom.getRowDimension(); r++) {
                    for (int c = 0; c < bottom.getColumnDimension(); c++) {
                        A_inv.setEntry(top.getRowDimension() + r, c, bottom.getEntry(r, c));
                    }
                }
            }
        }
        return A_inv;  // Возвращаем обратную матрицу
    }

    public static void main(String[] args) {
        // Исходные данные для матрицы A и вектора b
        double[][] matrixData = {
                {0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245},
                {0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101},
                {0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321},
                {0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183},
                {0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423},
                {0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923},
                {0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105}
        };
        double[] bData = {0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673};

        RealMatrix A = new Array2DRowRealMatrix(matrixData);  // Матрица A
        RealVector b = new ArrayRealVector(bData);  // Вектор b

        // Получение обратной матрицы и решение системы
        RealMatrix A_inv = getInv(A);
        RealVector x = A_inv.operate(b);  // Вектор решения x

        System.out.println("Решение системы Ax = b: " + x);

        // Сравнение с решением через LU-разложение
        DecompositionSolver solver = new LUDecomposition(A).getSolver();
        RealVector xLU = solver.solve(b);

        System.out.println("Решение системы Ax = b через Apache Commons Math LUDecomposition: " + xLU);
        System.out.println("Первая норма разности решения через LUDecomposition и нашего решения: " +
                x.subtract(xLU).getL1Norm());  // Норма разности решений
    }
}
