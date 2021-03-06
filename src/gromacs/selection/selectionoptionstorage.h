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
 * Declares gmx::SelectionOptionStorage.
 *
 * \author Teemu Murtola <teemu.murtola@cbr.su.se>
 * \ingroup module_selection
 */
#ifndef GMX_SELECTION_SELECTIONOPTIONSTORAGE_H
#define GMX_SELECTION_SELECTIONOPTIONSTORAGE_H

#include "../options/optionstoragetemplate.h"
#include "selectionenums.h"

namespace gmx
{

class Selection;
class SelectionOption;
class SelectionOptionAdjuster;

/*! \internal \brief
 * Converts, validates, and stores selection values.
 *
 * \ingroup module_selection
 */
class SelectionOptionStorage : public OptionStorageTemplate<Selection *>
{
    public:
        SelectionOptionStorage();
        virtual ~SelectionOptionStorage();

        /*! \brief
         * Initializes the storage from option settings.
         *
         * \param[in] settings   Storage settings.
         * \param[in] options    Options object.
         * \retval 0 on success.
         */
        int init(const SelectionOption &settings, Options *options);

        virtual const char *typeString() const { return "sel"; }
        virtual std::string formatValue(int i) const;

        /*! \brief
         * Adds selections to the storage.
         *
         * \param[in] selections  List of selections to add.
         * \param[in] bFullValue  If true, the provided selections are the full
         *      value of the option, and additional checks are performed.
         * \param[in] errors      Error reporter object.
         *
         * This function is used to implement the methods
         * SelectionCollection::parseRequestedFromStdin() and
         * SelectionCollection::parseRequestedFromString() (called with
         * \p bFullValue set to true), as well as internally by the storage
         * class (called with \p bFullValue set to false).
         */
        int addSelections(const std::vector<Selection *> &selections,
                          bool bFullValue, AbstractErrorReporter *errors);

        // Required to access the number of values in selection requests.
        // See SelectionCollection::Impl.
        using MyBase::maxValueCount;
        /*! \brief
         * Sets the number of selections allowed for this selection.
         *
         * \param[in] count       Required number of selections for this option.
         * \param[in] errors      Error reporter object.
         * \retval 0 on success.
         *
         * If values have already been provided, it is checked that a correct
         * number has been provided.  If requests have already been made, but
         * have not yet been processed, they are also affected.
         */
        int setAllowedValueCount(int count, AbstractErrorReporter *errors);
        /*! \brief
         * Alters flags for the selections created by this option.
         *
         * \param[in] flag        Flag to change.
         * \param[in] bSet        Whether to set or clear the flag.
         * \param[in] errors      Error reporter object.
         * \retval 0 on success.
         *
         * If values have already been provided, it is checked that they match
         * the limitations enforced by the flags.  If requests have already
         * been made, but have not yet been processed, they are also affected.
         */
        int setSelectionFlag(SelectionFlag flag, bool bSet,
                             AbstractErrorReporter *errors);

    private:
        virtual int convertValue(const std::string &value,
                                 AbstractErrorReporter *errors);
        virtual int processSet(int nvalues, AbstractErrorReporter *errors);
        virtual int processAll(AbstractErrorReporter *errors);

        SelectionFlags          _selectionFlags;
        //! Pointer to the adjuster (there can be only one, can be NULL).
        SelectionOptionAdjuster *_adjuster;
};

} // namespace gmx

#endif
