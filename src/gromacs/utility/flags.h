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
 *
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
/*! \libinternal \file
 * \brief
 * Declares gmx::FlagsTemplate.
 *
 * \author Teemu Murtola <teemu.murtola@cbr.su.se>
 * \inlibraryapi
 */
#ifndef GMX_UTILITY_FLAGS_H
#define GMX_UTILITY_FLAGS_H

namespace gmx
{

/*! \libinternal \brief
 * Template class for typesafe handling of combination of flags.
 *
 * \tparam T An enumerated type that holds the possible single flags.
 *
 * \inlibraryapi
 */
template <typename T>
class FlagsTemplate
{
    public:
        //! Creates a flags object with no flags set.
        FlagsTemplate() : _flags(0) {}
        //! Creates a flags object from a single flag.
        FlagsTemplate(T flag) : _flags(flag) {}

        //! Returns true if the given flag is set.
        bool test(T flag) const { return _flags & flag; }
        //! Clears all flags.
        void clearAll() { _flags = 0; }
        //! Sets the given flag.
        void set(T flag) { _flags |= flag; }
        //! Clears the given flag.
        void clear(T flag) { _flags &= ~flag; }
        //! Sets or clears the given flag.
        void set(T flag, bool bSet)
        {
            if (bSet)
            {
                set(flag);
            }
            else
            {
                clear(flag);
            }
        }

        //! Combines flags from two flags objects.
        FlagsTemplate<T> operator |(const FlagsTemplate<T> &other) const
        {
            return FlagsTemplate<T>(_flags | other._flags);
        }
        //! Combines flags from another flag object.
        FlagsTemplate<T> &operator |=(const FlagsTemplate<T> &other)
        {
            _flags |= other._flags;
            return *this;
        }

    private:
        //! Creates a flags object with the given flags.
        explicit FlagsTemplate(unsigned long flags) : _flags(flags) {}

        unsigned long           _flags;
};

} // namespace gmx

#endif
