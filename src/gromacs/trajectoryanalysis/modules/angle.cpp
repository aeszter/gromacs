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
#include "angle.h"

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <pbc.h>
#include <vec.h>

#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/errorreporting/abstracterrorreporter.h"
#include "gromacs/fatalerror/fatalerror.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/options.h"
#include "gromacs/selection/selection.h"
#include "gromacs/selection/selectionoption.h"

namespace gmx
{

namespace analysismodules
{

Angle::Angle()
    : _options("angle", "Angle calculation"),
      _sel1Adj(NULL), _sel2Adj(NULL),
      _bSplit1(false), _bSplit2(false), _bMulti(false), _bAll(false),
      _bDumpDist(false), _vt0(NULL)
{
}


Angle::~Angle()
{
    delete[] _vt0;
}


Options *
Angle::initOptions(TrajectoryAnalysisSettings *settings)
{
    static const char *const desc[] = {
        "g_angle computes different types of angles between vectors.",
        "It supports both vectors defined by two positions and normals of",
        "planes defined by three positions.",
        "The z axis or the local normal of a sphere can also be used as",
        "one of the vectors.",
        "There are also convenience options 'angle' and 'dihedral' for",
        "calculating bond angles and dihedrals defined by three/four",
        "positions.[PAR]",
        "The type of the angle is specified with [TT]-g1[tt] and [TT]-g2[tt].",
        "If [TT]-g1[tt] is [TT]angle[tt] or [TT]dihedral[tt], [TT]-g2[tt]",
        "should not be specified.",
        "In this case, one selection is required, and it should contain",
        "triplets or quartets of positions that define the angles to be",
        "calculated.",
        "If [TT]-g1[tt] is not [TT]angle[tt] or [TT]dihedral[tt], [TT]-g2[tt]",
        "should not be [TT]none[tt], and the two options define two vectors",
        "for the calculation. For vectors ([TT]vector[tt]), a selection with",
        "pairs of positions is required, and for planes ([TT]plane[tt]),",
        "triplets of positions are required.",
        "If both vectors are specified by positions, the number of vectors",
        "should be the same in both selections.",
        "[TT]-g2 sphnorm[tt] requires a reference selection that defines",
        "the center of the sphere.",
        "[TT]-g2 z[tt] does not require any selection.[PAR]",
        "With [TT]-split1[tt], the positions for [TT]-g1[tt] are specified",
        "using N separate selections with M positions each, instead of the",
        "default M*N positions in one selection.",
        "[TT]-split2[tt] does the same for [TT]-g2[tt].[PAR]",
        "There are two options for output:",
        "[TT]-o[tt] writes an xvgr file with the time and the average angle",
        "for each frame.",
        "With [TT]-all[tt], also the individual angles are written (only",
        "supported for static selections).",
        "[TT]-od[tt] can be used to dump all the individual angles,",
        "each on a separate line. This format is better suited for",
        "further processing, e.g., if angles from multiple runs are needed.",
        NULL
    };
    static const char *const cGroup1TypeEnum[] =
        { "angle", "dihedral", "vector", "plane", NULL };
    static const char *const cGroup2TypeEnum[] =
        { "none", "vector", "plane", "t0", "z", "sphnorm", NULL };

    _options.setDescription(desc);

    _options.addOption(FileNameOption("o").filetype(eftPlot).writeOnly()
                           .store(&_fnAngle).defaultValueIfSet("angle"));
    _options.addOption(FileNameOption("od").filetype(eftPlot).writeOnly()
                           .store(&_fnDump).defaultValueIfSet("angdump"));

    _options.addOption(StringOption("g1").enumValue(cGroup1TypeEnum)
        .defaultEnumIndex(0).store(&_g1type)
        .description("Type of analysis/first vector group"));
    _options.addOption(StringOption("g2").enumValue(cGroup2TypeEnum)
        .defaultEnumIndex(0).store(&_g2type)
        .description("Type of second vector group"));
    _options.addOption(BooleanOption("split1").store(&_bSplit1)
        .description("Each position of first group in separate selection"));
    _options.addOption(BooleanOption("split2").store(&_bSplit2)
        .description("Each position of second group in separate selection"));
    _options.addOption(BooleanOption("multi").store(&_bMulti)
        .description("Analyze multiple sets of angles/dihedrals"));
    _options.addOption(BooleanOption("all").store(&_bAll)
        .description("Print individual angles together with the average"));
    _options.addOption(BooleanOption("dumpd").store(&_bDumpDist)
        .description("Write also distances with -od"));

    _options.addOption(SelectionOption("group1").multiValue().required()
        .dynamicOnlyWhole().storeVector(&_sel1).getAdjuster(&_sel1Adj)
        .description("First analysis/vector selection"));
    _options.addOption(SelectionOption("group2").multiValue()
        .dynamicOnlyWhole().storeVector(&_sel2).getAdjuster(&_sel2Adj)
        .description("Second analysis/vector selection"));

    return &_options;
}


int
Angle::initOptionsDone(TrajectoryAnalysisSettings *settings,
                       AbstractErrorReporter *errors)
{
    // Validity checks.
    bool bSingle = (_g1type[0] == 'a' || _g1type[0] == 'd');

    if (bSingle && _g2type[0] != 'n')
    {
        errors->error("Cannot use a second group (-g2) with -g1 angle or dihedral");
        return eeInconsistentInput;
    }
    if (bSingle && _options.isSet("group2"))
    {
        errors->error("Cannot provide a second selection (-group2) with "
                      "-g1 angle or dihedral");
        return eeInconsistentInput;
    }
    if (!bSingle && _g2type[0] == 'n')
    {
        errors->error("Should specify a second group (-g2) if the first group "
                      "is not an angle or a dihedral");
        return eeInconsistentInput;
    }
    if (bSingle && _bDumpDist)
    {
        errors->warning("Cannot calculate distances with -g1 angle or dihedral");
        _bDumpDist = false;
    }
    if (_bMulti && _bSplit1)
    {
        errors->error("-mult can only be combined with -g1 angle or dihedral");
        return eeInconsistentInput;
    }
    if (!bSingle && _bMulti)
    {
        errors->error("-mult can only be combined with -g1 angle or dihedral");
        return eeInconsistentInput;
    }
    if (_bMulti && _bAll)
    {
        errors->error("-mult and -all are mutually exclusive options");
        return eeInconsistentInput;
    }
    if (_bAll)
    {
        int rc = _sel1Adj->setOnlyStatic(true);
        if (rc != 0)
        {
            return rc;
        }
    }

    // Set up the number of positions per angle.
    switch (_g1type[0])
    {
        case 'a': _natoms1 = 3; break;
        case 'd': _natoms1 = 4; break;
        case 'v': _natoms1 = 2; break;
        case 'p': _natoms1 = 3; break;
        default:
            GMX_ERROR(eeInternalError, "invalid -g1 value");
    }
    switch (_g2type[0])
    {
        case 'n': _natoms2 = 0; break;
        case 'v': _natoms2 = 2; break;
        case 'p': _natoms2 = 3; break;
        case 't': _natoms2 = 0; break;
        case 'z': _natoms2 = 0; break;
        case 's': _natoms2 = 1; break;
        default:
            GMX_ERROR(eeInternalError, "invalid -g2 value");
    }
    if (_natoms2 == 0 && _options.isSet("group2"))
    {
        errors->error("Cannot provide a second selection (-group2) with -g2 t0 or z");
        return eeInconsistentInput;
    }

    if (!_bMulti)
    {
        OptionAdjusterErrorContext context(_sel1Adj, errors);
        int rc = _sel1Adj->setValueCount(_bSplit1 ? _natoms1 : 1);
        if (rc != 0)
        {
            return rc;
        }
    }
    if (_natoms2 > 0)
    {
        OptionAdjusterErrorContext context(_sel2Adj, errors);
        int rc = _sel2Adj->setValueCount(_bSplit2 ? _natoms2 : 1);
        if (rc != 0)
        {
            return rc;
        }
    }

    return 0;
}


int
Angle::checkSelections(const std::vector<Selection *> &sel1,
                       const std::vector<Selection *> &sel2) const
{
    if (_bMulti)
    {
        for (size_t g = 0; g < sel1.size(); ++g)
        {
            if (sel1[g]->posCount() % _natoms1 != 0)
            {
                fatalErrorFormatted(eeInconsistentInput, GMX_ERRORLOC,
                    "Number of positions in selection %d not divisible by %d",
                    static_cast<int>(g + 1), _natoms1);
                return eeInconsistentInput;
            }
        }
        return 0;
    }

    int na1 = sel1[0]->posCount();
    int na2 = (_natoms2 > 0) ? sel2[0]->posCount() : 0;

    if (!_bSplit1 && _natoms1 > 1 && na1 % _natoms1 != 0)
    {
        fatalErrorFormatted(eeInconsistentInput, GMX_ERRORLOC,
            "Number of positions in the first group not divisible by %d",
            _natoms1);
        return eeInconsistentInput;
    }
    if (!_bSplit2 && _natoms2 > 1 && na2 % _natoms2 != 0)
    {
        fatalErrorFormatted(eeInconsistentInput, GMX_ERRORLOC,
            "Number of positions in the second group not divisible by %d",
            _natoms2);
        return eeInconsistentInput;
    }

    if (_bSplit1)
    {
        for (int g = 1; g < _natoms1; ++g)
        {
            if (sel1[g]->posCount() != na1)
            {
                GMX_ERROR(eeInconsistentInput,
                          "All selections in the first group should contain "
                          "the same number of positions");
            }
        }
    }
    else
    {
        na1 /= _natoms1;
    }
    if (_natoms2 > 1)
    {
        if (_bSplit2)
        {
            for (int g = 1; g < _natoms2; ++g)
            {
                if (sel2[g]->posCount() != na2)
                {
                    GMX_ERROR(eeInconsistentInput,
                              "All selections in the second group should contain "
                              "the same number of positions");
                }
            }
        }
        else
        {
            na2 /= _natoms2;
        }
    }
    if (_natoms1 > 0 && _natoms2 > 1 && na1 != na2)
    {
        GMX_ERROR(eeInconsistentInput,
                  "Number of vectors defined by the two groups are not the same");
    }
    if (_g2type[0] == 's' && sel2[0]->posCount() != 1)
    {
        GMX_ERROR(eeInconsistentInput,
                  "The second group should contain a single position with -g2 sphnorm");
    }
    return 0;
}


int
Angle::initAnalysis(const TopologyInformation &top)
{
    int rc = checkSelections(_sel1, _sel2);
    if (rc != 0)
    {
        return rc;
    }

    if (_bMulti)
    {
        _data.setColumns(_sel1.size());
    }
    else if (_bAll)
    {
        int na = _sel1[0]->posCount();
        if (!_bSplit1)
        {
            na /= _natoms1;
        }
        _data.setColumns(na + 1);
    }
    else
    {
        _data.setColumns(1);
    }

    if (_g2type == "t0")
    {
        int na = _sel1[0]->posCount();
        if (!_bSplit1)
        {
            na /= _natoms1;
        }
        _vt0 = new rvec[na];
    }

    registerAnalysisDataset(&_data, "angle");

    AnalysisDataPlotModule *plotm = new AnalysisDataPlotModule(_options);
    plotm->setFileName(_fnAngle);
    plotm->setTitle("Angle");
    plotm->setXTimeLabel();
    plotm->setYLabel("Angle [degrees]");
    _data.addModule(plotm);

    return 0;
}


static void
copy_pos(const std::vector<Selection *> &sel, bool bSplit, int natoms,
         int firstg, int first, rvec x[])
{
    if (bSplit)
    {
        for (int k = 0; k < natoms; ++k)
        {
            copy_rvec(sel[firstg + k]->x(first), x[k]);
        }
    }
    else
    {
        for (int k = 0; k < natoms; ++k)
        {
            copy_rvec(sel[firstg]->x(first + k), x[k]);
        }
    }
}


static void
calc_vec(int natoms, rvec x[], t_pbc *pbc, rvec xout, rvec cout)
{
    switch (natoms)
    {
        case 2:
            if (pbc)
            {
                pbc_dx(pbc, x[1], x[0], xout);
            }
            else
            {
                rvec_sub(x[1], x[0], xout);
            }
            svmul(0.5, xout, cout);
            rvec_add(x[0], cout, cout);
            break;
        case 3: {
            rvec v1, v2;
            if (pbc)
            {
                pbc_dx(pbc, x[1], x[0], v1);
                pbc_dx(pbc, x[2], x[0], v2);
            }
            else
            {
                rvec_sub(x[1], x[0], v1);
                rvec_sub(x[2], x[0], v2);
            }
            cprod(v1, v2, xout);
            rvec_add(x[0], x[1], cout);
            rvec_add(cout, x[2], cout);
            svmul(1.0/3.0, cout, cout);
            break;
        }
    }
}


int
Angle::analyzeFrame(int frnr, const t_trxframe &fr, t_pbc *pbc,
                    TrajectoryAnalysisModuleData *pdata)
{
    AnalysisDataHandle *dh = pdata->dataHandle("angle");
    std::vector<Selection *> sel1 = pdata->parallelSelections(_sel1);
    std::vector<Selection *> sel2 = pdata->parallelSelections(_sel2);

    int rc = checkSelections(sel1, sel2);
    if (rc != 0)
    {
        return rc;
    }

    rvec  v1, v2;
    rvec  c1, c2;
    switch (_g2type[0])
    {
        case 'z':
            clear_rvec(v2);
            v2[ZZ] = 1.0;
            clear_rvec(c2);
            break;
        case 's':
            copy_rvec(_sel2[0]->x(0), c2);
            break;
    }

    dh->startFrame(frnr, fr.time);

    int incr1 = _bSplit1 ? 1 : _natoms1;
    int incr2 = _bSplit2 ? 1 : _natoms2;
    int ngrps = _bMulti ? _sel1.size() : 1;

    for (int g = 0; g < ngrps; ++g)
    {
        real ave = 0.0;
        int n = 0;
        int i, j;
        for (i = j = 0; i < sel1[g]->posCount(); i += incr1)
        {
            rvec x[4];
            real angle;
            copy_pos(sel1, _bSplit1, _natoms1, g, i, x);
            switch (_g1type[0])
            {
                case 'a':
                    if (pbc)
                    {
                        pbc_dx(pbc, x[0], x[1], v1);
                        pbc_dx(pbc, x[2], x[1], v2);
                    }
                    else
                    {
                        rvec_sub(x[0], x[1], v1);
                        rvec_sub(x[2], x[1], v2);
                    }
                    angle = gmx_angle(v1, v2);
                    break;
                case 'd': {
                    rvec dx[3];
                    if (pbc)
                    {
                        pbc_dx(pbc, x[0], x[1], dx[0]);
                        pbc_dx(pbc, x[2], x[1], dx[1]);
                        pbc_dx(pbc, x[2], x[3], dx[2]);
                    }
                    else
                    {
                        rvec_sub(x[0], x[1], dx[0]);
                        rvec_sub(x[2], x[1], dx[1]);
                        rvec_sub(x[2], x[3], dx[2]);
                    }
                    cprod(dx[0], dx[1], v1);
                    cprod(dx[1], dx[2], v2);
                    angle = gmx_angle(v1, v2);
                    real ipr = iprod(dx[0], v2);
                    if (ipr < 0)
                    {
                        angle = -angle;
                    }
                    break;
                }
                case 'v':
                case 'p':
                    calc_vec(_natoms1, x, pbc, v1, c1);
                    switch (_g2type[0])
                    {
                        case 'v':
                        case 'p':
                            copy_pos(sel2, _bSplit2, _natoms2, 0, j, x);
                            calc_vec(_natoms2, x, pbc, v2, c2);
                            j += incr2;
                            break;
                        case 't':
                            // FIXME: This is not parallelizable.
                            if (frnr == 0)
                            {
                                copy_rvec(v1, _vt0[n]);
                            }
                            copy_rvec(_vt0[n], v2);
                            break;
                        case 'z':
                            c1[XX] = c1[YY] = 0.0;
                            break;
                        case 's':
                            if (pbc)
                            {
                                pbc_dx(pbc, c1, c2, v2);
                            }
                            else
                            {
                                rvec_sub(c1, c2, v2);
                            }
                            break;
                        default:
                            GMX_ERROR(eeInternalError, "invalid -g2 value");
                    }
                    angle = gmx_angle(v1, v2);
                    break;
                default:
                    GMX_ERROR(eeInternalError, "invalid -g1 value");
            }
            angle *= RAD2DEG;
            real dist = 0.0;
            if (_bDumpDist)
            {
                if (pbc)
                {
                    rvec dx;
                    pbc_dx(pbc, c2, c1, dx);
                    dist = norm(dx);
                }
                else
                {
                    dist = sqrt(distance2(c1, c2));
                }
            }
            if (_bAll)
            {
                dh->addPoint(n + 1, angle);
            }
            ave += angle;
            ++n;
        }
        if (n > 0)
        {
            ave /= n;
        }
        dh->addPoint(g, ave);
    }
    dh->finishFrame();
    return 0;
}


int
Angle::finishAnalysis(int /*nframes*/)
{
    return 0;
}


int
Angle::writeOutput()
{
    return 0;
}


TrajectoryAnalysisModule *
Angle::create()
{
    return new Angle();
}

} // namespace modules

} // namespace gmxana
