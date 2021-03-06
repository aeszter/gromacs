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
 * Declares gmx::CommandLineParser.
 *
 * \author Teemu Murtola <teemu.murtola@cbr.su.se>
 * \inpublicapi
 * \ingroup module_options
 */
#ifndef GMX_OPTIONS_CMDLINEPARSER_H
#define GMX_OPTIONS_CMDLINEPARSER_H

namespace gmx
{

class AbstractErrorReporter;

class Options;

/*! \brief
 * Implements command-line parsing for Options objects.
 *
 * Typical usage (without error checking):
 * \code
gmx::Options options("name", "description");
// Fill up options

gmx::StandardErrorReporter errors;
gmx::CommandLineParser(&options, &errors).parse(&argc, argv);
options.finish(&errors);
 * \endcode
 *
 * \inpublicapi
 * \ingroup module_options
 */
class CommandLineParser
{
    public:
        /*! \brief
         * Creates a command-line parser that sets values for options.
         *
         * \param[in] options  Options object whose options should be set.
         * \param[in] errors   Error reporter object.
         */
        CommandLineParser(Options *options, AbstractErrorReporter *errors);
        ~CommandLineParser();

        /*! \brief
         * Parses the command-line.
         *
         * \retval 0 if there were no errors.
         */
        int parse(int *argc, char *argv[]);

    private:
        class Impl;

        Impl                   *_impl;

        // Disallow copy and assign.
        CommandLineParser(const CommandLineParser &);
        void operator =(const CommandLineParser &);
};

} // namespace gmx

#endif
