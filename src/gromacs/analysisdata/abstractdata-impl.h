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
 * Declares internal implementation classes for gmx::AbstractAnalysisData and
 * gmx::AbstractAnalysisDataStored.
 *
 * \author Teemu Murtola <teemu.murtola@cbr.su.se>
 * \ingroup module_analysisdata
 */
#ifndef GMX_ANALYSISDATA_ABSTRACTDATA_IMPL_H
#define GMX_ANALYSISDATA_ABSTRACTDATA_IMPL_H

#include <vector>

#include "types/simple.h"
#include "abstractdata.h"

namespace gmx
{

/*! \internal \brief
 * Private implementation class for AbstractAnalysisData.
 *
 * \ingroup module_analysisdata
 */
class AbstractAnalysisData::Impl
{
    public:
        //! Shorthand for list of modules added to the data.
        typedef std::vector<AnalysisDataModuleInterface *> ModuleList;

        Impl();
        ~Impl();

        /*! \brief
         * Present data already added to the data object to a module.
         *
         * \param[in] data   Data object to read data from.
         * \param[in] module Module to present the data to.
         * \retval ::eeInvalidValue if \p module is not compatible with the
         *      data object.
         * \retval ::eedataDataNotAvailable if all data is not  available
         *      through getData().
         *
         * Uses getData() in \p data to access all data in the object, and
         * calls the notification functions in \p module as if the module had
         * been registered to the data object when the data was added.
         */
        int presentData(AbstractAnalysisData *data,
                        AnalysisDataModuleInterface *module);

        //! List of modules added to the data.
        ModuleList              _modules;
        //! Whether notifyDataStart() has been called.
        bool                    _bDataStart;
        //! Whether new data is being added.
        bool                    _bInData;
        //! Whether data for a frame is being added.
        bool                    _bInFrame;
        //! true if all modules support missing data.
        bool                    _bAllowMissing;
        //! x value for the current frame.
        real                    _currx;
        //! dx value for the current frame.
        real                    _currdx;
};

/*! \internal \brief
 * Internal implementation class for storing a single data frame.
 *
 * \ingroup module_analysisdata
 */
class AnalysisDataFrame
{
    public:
        AnalysisDataFrame();
        ~AnalysisDataFrame();

        //! Allocate memory for a given number of columns.
        void allocate(int ncol);

        //! Zero-based global index of the frame.
        int                     _index;
        //! x value of the frame.
        real                    _x;
        //! Error of x for the frame.
        real                    _dx;
        //! Array of column values for the frame.
        real                   *_y;
        //! Array of column error values for the frame.
        real                   *_dy;
        //! Array of flags that tell whether a value is present.
        bool                   *_present;
};

/*! \internal \brief
 * Private implementation class for AbstractAnalysisDataStored.
 *
 * \ingroup module_analysisdata
 */
class AbstractAnalysisDataStored::Impl
{
    public:
        //! Shorthand for a list of data frames that are currently stored.
        typedef std::vector<AnalysisDataFrame *> FrameList;

        Impl();
        ~Impl();

        /*! \brief
         * Calculates the index of a frame in the storage vector.
         *
         * \param[in] index  Zero-based index for the frame to query.
         *      Negative value counts backwards from the current frame.
         * \returns Index in \a _store corresponding to \p index,
         *      or -1 if not available.
         */
        int getStoreIndex(int index) const;

        /*! \brief
         * Total number of complete frames in the data.
         */
        int                     _nframes;
        /*! \brief
         * Number of elements in \a _store.
         *
         * Also holds the number of frames that should be stored, even before
         * \a _store has been allocated.
         */
        int                     _nalloc;
        //! Whether all frames should be stored.
        bool                    _bStoreAll;
        //! List of data frames that are currently stored.
        FrameList               _store;
        /*! \brief
         * Index in \a _store where the next frame will be stored.
         *
         * This counter is incremented after notifyPointsAdd() has been called
         * for the frame.
         */
        int                     _nextind;
};

} // namespace gmx

#endif
