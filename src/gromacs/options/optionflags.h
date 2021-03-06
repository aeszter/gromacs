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
 * Defines flags used in option implementation.
 *
 * Symbols in this header are considered an implementation detail, and should
 * not be accessed outside the module.
 * Because of details in the implementation, it is still installed.
 *
 * \author Teemu Murtola <teemu.murtola@cbr.su.se>
 * \ingroup module_options
 */
#ifndef GMX_OPTIONS_OPTIONFLAGS_H
#define GMX_OPTIONS_OPTIONFLAGS_H

#include "../utility/flags.h"

namespace gmx
{

/*! \internal \brief
 * Flags for options.
 *
 * These flags are not part of the public interface, even though they are in an
 * installed header.  They are needed in a few template class implementations.
 */
enum OptionFlag
{
    //! %Option has been set.
    efSet                 = 1<<0,
    /*! \brief
     * The current value of the option is a default value.
     *
     * This flag is also set when a new option source starts, such that values
     * from the new source will overwrite old ones.
     */
    efHasDefaultValue     = 1<<1,
    //! %Option is required to be set.
    efRequired            = 1<<2,
    //! %Option can be specified multiple times.
    efMulti               = 1<<3,
    //! %Option is hidden from standard help.
    efHidden              = 1<<4,
    /*! \brief
     * %Option provides a boolean value.
     *
     * This is used to optionally support an alternative syntax where an
     * option provided with no value sets the value to true and an
     * option prefixed with "no" clears the value.
     */
    efBoolean             = 1<<5,
    /*! \brief
     * %Option value is a vector, but a single value is also accepted.
     *
     * If only a single value is provided, the storage object should fill the
     * whole vector with that value.  The length of the vector must be fixed.
     * The default length is 3 elements.
     */
    efVector              = 1<<6,
    efExternalStore       = 1<<8,
    efExternalStoreArray  = 1<<9,
    efExternalValueVector = 1<<10,
    //! %Option does not support default values.
    efNoDefaultValue      = 1<<7,
    /*! \brief
     * Storage object may add zero values even when a value is provided.
     *
     * In order to do proper error checking, this flag should be set when it is
     * possible that the AbstractOptionStorage::appendValue() method of the
     * storage object does not add any values for the option and still
     * succeeds.
     */
    efConversionMayNotAddValues = 1<<11,
    /*! \brief
     * Storage object does its custom checking for minimum value count.
     *
     * If this flag is set, the class derived from AbstractOptionStorage should
     * implement processSet(), processAll(), and possible other functions it
     * provides such that it always fails if not enough values are provided.
     * This is useful to override the default check, which is done in
     * AbstractOptionStorage::processSet().
     */
    efDontCheckMinimumCount     = 1<<16,
    efFile                = 1<<12,
    efFileRead            = 1<<13,
    efFileWrite           = 1<<14,
    efFileLibrary         = 1<<15,
    //efDynamic             = 1<<16,
    //efRanges              = 1<<17,
    //efEnum                = 1<<18,
    //efStaticEnum          = 1<<19,
    //efVarNum              = 1<<20,
    //efAtomVal             = 1<<21,
};

//! Holds a combination of ::OptionFlag values.
typedef FlagsTemplate<OptionFlag> OptionFlags;

} // namespace gmx

#endif
