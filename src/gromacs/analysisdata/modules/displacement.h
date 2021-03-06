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
 * Declares gmx::AnalysisDataDisplacementModule.
 *
 * \author Teemu Murtola <teemu.murtola@cbr.su.se>
 * \inpublicapi
 * \ingroup module_analysisdata
 */
#ifndef GMX_ANALYSISDATA_MODULES_DISPLACEMENT_H
#define GMX_ANALYSISDATA_MODULES_DISPLACEMENT_H

#include "../analysisdata.h"
#include "../datamodule.h"

namespace gmx
{

class AnalysisDataBinAverageModule;

/*! \brief
 * Data module for calculating displacements.
 *
 * Output data contains a frame for each frame in the input data except the
 * first one.  For each frame, there can be multiple points, each of which
 * describes displacement for a certain time difference ending that that frame.
 * The first column contains the time difference (backwards from the current
 * frame), and the remaining columns the sizes of the displacements.
 *
 * Current implementation is not very generic, but should be easy to extend.
 *
 * \inpublicapi
 * \ingroup module_analysisdata
 */
class AnalysisDataDisplacementModule : public AbstractAnalysisData,
                                       public AnalysisDataModuleInterface
{
    public:
        AnalysisDataDisplacementModule();
        virtual ~AnalysisDataDisplacementModule();

        /*! \brief
         * Sets the largest displacement time to be calculated.
         */
        void setMaxTime(real tmax);
        /*! \brief
         * Sets an histogram module that will receive a MSD histogram.
         *
         * If this function is not called, no histogram is calculated.
         */
        int setMSDHistogram(AnalysisDataBinAverageModule *histm);

        virtual int frameCount() const;
        virtual int getDataWErr(int index, real *x, real *dx,
                                const real **y, const real **dy,
                                const bool **present = 0) const;
        virtual int requestStorage(int nframes = -1);

        virtual int flags() const;

        virtual int dataStarted(AbstractAnalysisData *data);
        virtual int frameStarted(real x, real dx);
        virtual int pointsAdded(real x, real dx, int firstcol, int n,
                                const real *y, const real *dy,
                                const bool *present);
        virtual int frameFinished();
        virtual int dataFinished();

    private:
        class Impl;

        Impl                   *_impl;
};

} // namespace gmx

#endif
