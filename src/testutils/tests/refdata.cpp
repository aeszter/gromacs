/*
 *
 *                This source code is part of
 *
 *                 G   R   O   M   A   C   S
 *
 *          GROningen MAchine for Chemical Simulations
 *
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2009, The GROMACS development team,
 * check out http://www.gromacs.org for more information.

 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * If you want to redistribute modifications, please consider that
 * scientific software is very special. Version control is crucial -
 * bugs must be traceable. We will be happy to consider code for
 * inclusion in the official distribution, but derived work must not
 * be called official GROMACS. Details are found in the README & COPYING
 * files - if they are missing, get the official version at www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the papers on the package - you can find them in the top README file.
 *
 * For more info, check our website at http://www.gromacs.org
 */
/*! \internal \file
 * \brief
 * Tests utilities for test reference data.
 *
 * \author Teemu Murtola <teemu.murtola@cbr.su.se>
 * \ingroup module_testutils
 */
#include <vector>

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include "testutils/refdata.h"

namespace
{

TEST(ReferenceDataTest, HandlesSimpleData)
{
    using gmx::test::TestReferenceData;

    {
        TestReferenceData data(gmx::test::erefdataUpdateAll);
        ASSERT_NO_FATAL_FAILURE(data.checkBoolean(true, "int"));
        ASSERT_NO_FATAL_FAILURE(data.checkInteger(1, "int"));
        ASSERT_NO_FATAL_FAILURE(data.checkDouble(0.5, "real"));
        ASSERT_NO_FATAL_FAILURE(data.checkString("Test", "string"));
    }
    {
        TestReferenceData data(gmx::test::erefdataCompare);
        ASSERT_NO_FATAL_FAILURE(data.checkBoolean(true, "int"));
        ASSERT_NO_FATAL_FAILURE(data.checkInteger(1, "int"));
        ASSERT_NO_FATAL_FAILURE(data.checkDouble(0.5, "real"));
        ASSERT_NO_FATAL_FAILURE(data.checkString("Test", "string"));
    }
}


TEST(ReferenceDataTest, HandlesVectorData)
{
    using gmx::test::TestReferenceData;
    int veci[3] = { -1, 3, 5 };
    float vecf[3] = { -2.3, 1.43, 2.5 };
    double vecd[3] = { -2.3, 1.43, 2.5 };

    {
        TestReferenceData data(gmx::test::erefdataUpdateAll);
        ASSERT_NO_FATAL_FAILURE(data.checkVector(veci, "ivec"));
        ASSERT_NO_FATAL_FAILURE(data.checkVector(vecf, "fvec"));
        ASSERT_NO_FATAL_FAILURE(data.checkVector(vecd, "dvec"));
    }
    {
        TestReferenceData data(gmx::test::erefdataCompare);
        ASSERT_NO_FATAL_FAILURE(data.checkVector(veci, "ivec"));
        ASSERT_NO_FATAL_FAILURE(data.checkVector(vecf, "fvec"));
        ASSERT_NO_FATAL_FAILURE(data.checkVector(vecd, "dvec"));
    }
}


TEST(ReferenceDataTest, HandlesSequenceData)
{
    using gmx::test::TestReferenceData;
    int seq[5] = { -1, 3, 5, 2, 4 };

    {
        TestReferenceData data(gmx::test::erefdataUpdateAll);
        ASSERT_NO_FATAL_FAILURE(data.checkSequenceInteger(5, seq, "seq"));
    }
    {
        TestReferenceData data(gmx::test::erefdataCompare);
        ASSERT_NO_FATAL_FAILURE(data.checkSequenceInteger(5, seq, "seq"));
    }
}



TEST(ReferenceDataTest, HandlesIncorrectData)
{
    using gmx::test::TestReferenceData;

    {
        TestReferenceData data(gmx::test::erefdataUpdateAll);
        ASSERT_NO_FATAL_FAILURE(data.checkInteger(1, "int"));
        ASSERT_NO_FATAL_FAILURE(data.checkDouble(0.5, "real"));
        ASSERT_NO_FATAL_FAILURE(data.checkString("Test", "string"));
    }
    {
        TestReferenceData data(gmx::test::erefdataCompare);
        EXPECT_NONFATAL_FAILURE(data.checkInteger(2, "int"), "");
        EXPECT_NONFATAL_FAILURE(data.checkDouble(0.3, "real"), "");
        EXPECT_NONFATAL_FAILURE(data.checkString("Test2", "string"), "");
    }
}


TEST(ReferenceDataTest, HandlesMissingReferenceData)
{
    using gmx::test::TestReferenceData;

    EXPECT_FATAL_FAILURE(TestReferenceData data(gmx::test::erefdataCompare), "");
}


TEST(ReferenceDataTest, HandlesSpecialCharactersInStrings)
{
    using gmx::test::TestReferenceData;

    {
        TestReferenceData data(gmx::test::erefdataUpdateAll);
        ASSERT_NO_FATAL_FAILURE(data.checkString("\"<'>\n \r &\\/;", "string"));
    }
    {
        TestReferenceData data(gmx::test::erefdataCompare);
        ASSERT_NO_FATAL_FAILURE(data.checkString("\"<'>\n \r &\\/;", "string"));
    }
}

} // namespace
