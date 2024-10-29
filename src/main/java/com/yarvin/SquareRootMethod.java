package com.yarvin;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Locale;

public class SquareRootMethod {

    public static Complex[] squareRootMethod(double[][] A, double[] b) {
        int n = A.length;
        Complex[][] B = new Complex[n][n];
        Complex[] y = new Complex[n];
        Complex[] x = new Complex[n];

        // Инициализация массивов B, y и x
        for (int i = 0; i < n; i++) {
            y[i] = Complex.ZERO;
            x[i] = Complex.ZERO;
            for (int j = 0; j < n; j++) {
                B[i][j] = Complex.ZERO;
            }
        }

        // Вычисляем матрицу B и вектор y
        for (int i = 0; i < n; i++) {
            Complex sumBDiag = Complex.ZERO;
            for (int k = 0; k < i; k++) {
                sumBDiag = sumBDiag.add(B[k][i].multiply(B[k][i]));
            }
            Complex diagonalValue = new Complex(A[i][i], 0.0).subtract(sumBDiag);

            // Проверка на положительность диагонального элемента
            if (diagonalValue.getReal() < 0) {
                diagonalValue = new Complex(diagonalValue.getReal(), 0.0);
            }
            B[i][i] = diagonalValue.sqrt();

            // Вычисление наддиагональных элементов B
            for (int j = i + 1; j < n; j++) {
                Complex sumBOffDiag = Complex.ZERO;
                for (int k = 0; k < i; k++) {
                    sumBOffDiag = sumBOffDiag.add(B[k][i].multiply(B[k][j]));
                }
                B[i][j] = new Complex(A[i][j], 0.0).subtract(sumBOffDiag).divide(B[i][i]);
            }

            // Вычисление элементов вектора y
            Complex sumY = Complex.ZERO;
            for (int k = 0; k < i; k++) {
                sumY = sumY.add(B[k][i].multiply(y[k]));
            }
            y[i] = new Complex(b[i], 0.0).subtract(sumY).divide(B[i][i]);
        }

        // Вычисление вектора x (обратная подстановка)
        for (int i = n - 1; i >= 0; i--) {
            Complex sumX = Complex.ZERO;
            for (int k = i + 1; k < n; k++) {
                sumX = sumX.add(B[i][k].multiply(x[k]));
            }
            x[i] = y[i].subtract(sumX).divide(B[i][i]);
        }

        return x;
    }

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);

        double[][] A = {
                {2.2, 4, -3, 1.5, 0.6, 2, 0.7},
                {4, 3.2, 1.5, -0.7, -0.8, 3, 1},
                {-3, 1.5, 1.8, 0.9, 3, 2, 2},
                {1.5, -0.7, 0.9, 2.2, 4, 3, 1},
                {0.6, -0.8, 3, 4, 3.2, 0.6, 0.7},
                {2, 3, 2, 3, 0.6, 2.2, 4},
                {0.7, 1, 2, 1, 0.7, 4, 3.2}
        };
        double[] b = {3.2, 4.3, -0.1, 3.5, 5.3, 9.0, 3.7};

        Complex[] x = squareRootMethod(A, b);


        System.out.print("Ответ: [");
        for (int i = 0; i < x.length; i++) {
            System.out.printf(" %.8f+%.1fj", x[i].getReal(), x[i].getImaginary());
            if (i < x.length - 1) {
                System.out.print(",");
            }
        }
        System.out.println(" ]");


        RealMatrix matrixA = new Array2DRowRealMatrix(A);
        RealVector vectorB = new ArrayRealVector(b);
        DecompositionSolver solver = new LUDecomposition(matrixA).getSolver();
        RealVector result = solver.solve(vectorB);

        // Преобразование массива x к действительным значениям для проверки разницы
        double[] realPartOfX = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            realPartOfX[i] = x[i].getReal();
        }

        RealVector realVectorX = new ArrayRealVector(realPartOfX);
        double normDiff = result.subtract(realVectorX).getNorm();
        System.out.printf("Первая норма разности решения через Apache Commons Math и нашего решения: %.15e%n", normDiff);

        // Вычисляем норму для произведения матрицы A на вектор x и проверки
        double normAxMinusB = matrixA.operate(result).subtract(vectorB).getNorm();
        System.out.printf("Первая норма разности произведения матрицы A на вектор x и вектора b: %.15e%n", normAxMinusB);
    }
}



