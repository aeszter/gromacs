#include "gmxTestHarness.h"
extern "C" {
#include "gmx_matrix.h"
}

TEST(MatrixTest, MultiplyInverse)
{
  double **m1, **m2, **product;
  int ret;
  int i, j;

  m1 = alloc_matrix (3, 3);
  m2 = alloc_matrix (3, 3);
  product = alloc_matrix (3, 3);

  m1 [0][0] = 1;
  m1 [0][1] = 2;
  m1 [0][2] = 3;
  m1 [1][0] = 4;
  m1 [1][1] = 5;
  m1 [1][2] = 6;
  m1 [2][0] = 7;
  m1 [2][1] = 8;
  m1 [2][2] = 2;

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      m2[i][j] = m1[i][j];

  ret = matrix_invert (NULL, 3, m1);
  EXPECT_EQ (0, ret);

  matrix_multiply (NULL, 3, 3, m1, m2, product);

  ASSERT_NEAR (1, product [0][0], 1e-8);
  ASSERT_NEAR (1, product [1][1], 1e-8);
  ASSERT_NEAR (1, product [2][2], 1e-8);
  ASSERT_NEAR (0, product [0][1], 1e-8);
  ASSERT_NEAR (0, product [0][2], 1e-8);
  ASSERT_NEAR (0, product [1][0], 1e-8);
  ASSERT_NEAR (0, product [1][2], 1e-8);
  ASSERT_NEAR (0, product [2][0], 1e-8);
  ASSERT_NEAR (0, product [2][1], 1e-8);

  free_matrix (m1, 3);
  free_matrix (m2, 3);
  free_matrix (product, 3);

}
