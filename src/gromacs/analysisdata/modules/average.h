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
/*! \file
 * \brief
 * Declares gmx::AnalysisDataAverageModule.
 *
 * \author Teemu Murtola <teemu.murtola@cbr.su.se>
 * \inpublicapi
 * \ingroup module_analysisdata
 */
#ifndef GMX_ANALYSISDATA_MODULES_AVERAGE_H
#define GMX_ANALYSISDATA_MODULES_AVERAGE_H

#include "../arraydata.h"
#include "../datamodule.h"

namespace gmx
{

/*! \brief
 * Data module for simple averaging of columns.
 *
 * Output data contains a frame for each column of input data.
 * There are two columns: the average and standard deviation of
 * that column.
 * The data becomes available only after the original data has been
 * finished.
 *
 * Multipoint data and missing data points are both supported. The average
 * is always calculated over all data points present in a column.
 *
 * \inpublicapi
 * \ingroup module_analysisdata
 */
class AnalysisDataAverageModule : public AbstractAnalysisArrayData,
                                  public AnalysisDataModuleInterface
{
    public:
        AnalysisDataAverageModule();
        virtual ~AnalysisDataAverageModule();

        using AbstractAnalysisArrayData::setXAxis;

        virtual int flags() const;

        virtual int dataStarted(AbstractAnalysisData *data);
        virtual int frameStarted(real x, real dx);
        virtual int pointsAdded(real x, real dx, int firstcol, int n,
                                const real *y, const real *dy,
                                const bool *present);
        virtual int frameFinished();
        virtual int dataFinished();

        //! Convenience access to the average of a data column.
        real average(int index) const;
        //! Convenience access to the standard deviation of a data column.
        real stddev(int index) const;

    private:
        int                    *_nsamples;

        // Copy and assign disallowed by base.
};

} // namespace gmx

#endif
