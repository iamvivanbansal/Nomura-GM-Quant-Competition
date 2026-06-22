// ============================================================
// Interest Rate Curve Construction System
// C++17 | Clang and GCC Compatible
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <numeric>
#include <iomanip>
#include <cctype>

// ============================================================
// UTILITIES & SANITIZATION
// ============================================================

static double dcf(double t2, double t1 = 0.0) {
    return (t2 - t1) / 360.0;
}

// Safely strips carriage returns, tabs, and spaces from CSV inputs
static std::string sanitizeString(std::string s) {
    s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
    s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
    return s;
}

static int maturityToDays(const std::string& s) {
    std::string clean = sanitizeString(s);
    if (clean.empty()) throw std::invalid_argument("Empty maturity string");
    char suffix = clean.back();
    int val = std::stoi(clean.substr(0, clean.size() - 1));
    switch (suffix) {
        case 'D': return val;
        case 'W': return val * 7;
        case 'M': return val * 30;
        case 'Y': return val * 360;
        default:  throw std::invalid_argument("Unknown maturity suffix: " + clean);
    }
}

static int freqToDays(const std::string& s) {
    std::string clean = sanitizeString(s);
    int months = std::stoi(clean.substr(0, clean.size() - 1));
    return months * 30;
}

// Brent's method root finder
static double brent(std::function<double(double)> f, double lo, double hi,
                    double tol = 1e-12, int maxIter = 200) {
    double fa = f(lo), fb = f(hi);
    if (fa * fb > 0.0)
        throw std::runtime_error("brent: f(lo) and f(hi) must have opposite signs");
    if (std::abs(fa) < tol) return lo;
    if (std::abs(fb) < tol) return hi;

    double c = lo, fc = fa, d = hi - lo, e = d;
    for (int i = 0; i < maxIter; ++i) {
        if (fb * fc > 0.0) { c = lo; fc = fa; d = e = hi - lo; }
        if (std::abs(fc) < std::abs(fb)) {
            lo = hi; fa = fb; hi = c; fb = fc; c = lo; fc = fa;
        }
        double tol1 = 2.0 * 1e-15 * std::abs(hi) + 0.5 * tol;
        double xm = 0.5 * (c - hi);
        if (std::abs(xm) <= tol1 || std::abs(fb) < tol) return hi;
        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            double s = fb / fa, p, q, r;
            if (lo == c) {
                p = 2.0 * xm * s; q = 1.0 - s;
            } else {
                q = fa / fc; r = fb / fc;
                p = s * (2.0 * xm * q * (q - r) - (hi - lo) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q; else p = -p;
            if (2.0 * p < std::min(3.0 * xm * q - std::abs(tol1 * q), std::abs(e * q))) {
                e = d; d = p / q;
            } else { d = xm; e = d; }
        } else { d = xm; e = d; }
        lo = hi; fa = fb;
        hi += (std::abs(d) > tol1) ? d : (xm > 0 ? tol1 : -tol1);
        fb = f(hi);
    }
    return hi;
}

// Solve 3x3 linear system Ax=b via Cramer's rule
static std::vector<double> solve3x3(const double A[3][3], const double b[3]) {
    auto det3 = [](double a00, double a01, double a02,
                   double a10, double a11, double a12,
                   double a20, double a21, double a22) {
        return a00*(a11*a22-a12*a21) - a01*(a10*a22-a12*a20) + a02*(a10*a21-a11*a20);
    };
    double D = det3(A[0][0],A[0][1],A[0][2],
                    A[1][0],A[1][1],A[1][2],
                    A[2][0],A[2][1],A[2][2]);
    if (std::abs(D) < 1e-20) throw std::runtime_error("Singular 3x3 system");
    std::vector<double> x(3);
    for (int col = 0; col < 3; ++col) {
        double M[3][3];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                M[r][c] = (c == col) ? b[r] : A[r][c];
        x[col] = det3(M[0][0],M[0][1],M[0][2],
                      M[1][0],M[1][1],M[1][2],
                      M[2][0],M[2][1],M[2][2]) / D;
    }
    return x;
}

// ============================================================
// INTERPOLATION INTERFACE (Strategy Pattern)
// ============================================================

struct CurveNode {
    double t;
    double rate;
};

class InterpolationMethod {
public:
    virtual ~InterpolationMethod() = default;

    virtual double interpolate(double t,
                               const std::vector<double>& nodes_t,
                               const std::vector<double>& nodes_df) const = 0;

    virtual double d_lnDF_dDFk(double t, int k,
                               const std::vector<double>& nodes_t,
                               const std::vector<double>& nodes_df) const = 0;

    double getDF(double t,
                 const std::vector<double>& nodes_t,
                 const std::vector<double>& nodes_df) const {
        if (nodes_t.empty()) return 1.0;
        if (t < 1e-12) return 1.0;
        return interpolate(t, nodes_t, nodes_df);
    }

    double dDF_dDFk(double t, int k,
                    const std::vector<double>& nodes_t,
                    const std::vector<double>& nodes_df) const {
        double df_t = getDF(t, nodes_t, nodes_df);
        return df_t * d_lnDF_dDFk(t, k, nodes_t, nodes_df);
    }
};

// ============================================================
// LOG-LINEAR INTERPOLATION LAYER
// ============================================================

class LinearInterp : public InterpolationMethod {
public:
    double interpolate(double t,
                       const std::vector<double>& nt,
                       const std::vector<double>& nd) const override {
        int n = (int)nt.size();
        std::vector<double> ldf(n);
        for (int i = 0; i < n; ++i) ldf[i] = std::log(nd[i]);

        if (t <= nt[0])
            return std::exp(ldf[0] * t / nt[0]);
        if (t >= nt[n-1]) {
            double slope = (ldf[n-1] - ldf[n-2]) / (nt[n-1] - nt[n-2]);
            return std::exp(ldf[n-1] + slope * (t - nt[n-1]));
        }
        for (int i = 1; i < n; ++i) {
            if (t <= nt[i]) {
                double alpha = (t - nt[i-1]) / (nt[i] - nt[i-1]);
                return std::exp(ldf[i-1] + alpha * (ldf[i] - ldf[i-1]));
            }
        }
        throw std::runtime_error("LinearInterp: unreachable state");
    }

    double d_lnDF_dDFk(double t, int k,
                        const std::vector<double>& nt,
                        const std::vector<double>& nd) const override {
        int n = (int)nt.size();

        if (t <= nt[0])
            return (k == 0) ? (t / nt[0]) / nd[0] : 0.0;

        if (t >= nt[n-1]) {
            if (k == n-1) return (1.0 + (t - nt[n-1]) / (nt[n-1] - nt[n-2])) / nd[n-1];
            if (k == n-2) return (-(t - nt[n-1]) / (nt[n-1] - nt[n-2])) / nd[n-2];
            return 0.0;
        }
        for (int i = 1; i < n; ++i) {
            if (t <= nt[i]) {
                double alpha = (t - nt[i-1]) / (nt[i] - nt[i-1]);
                if (k == i-1) return (1.0 - alpha) / nd[i-1];
                if (k == i)   return alpha / nd[i];
                return 0.0;
            }
        }
        return 0.0;
    }
};

// ============================================================
// AVERAGED QUADRATIC INTERPOLATION LAYER
// ============================================================

class AvgQuadInterp : public InterpolationMethod {
public:
    static std::array<double,3> quadCoeffs(double ta, double la,
                                           double tb, double lb,
                                           double tc, double lc) {
        double A[3][3] = {{ta*ta,ta,1},{tb*tb,tb,1},{tc*tc,tc,1}};
        double b[3]    = {la, lb, lc};
        auto v = solve3x3(A, b);
        return {v[0], v[1], v[2]};
    }

    static double evalQuad(const std::array<double,3>& c, double t) {
        return c[0]*t*t + c[1]*t + c[2];
    }

    double interpolate(double t,
                       const std::vector<double>& nt,
                       const std::vector<double>& nd) const override {
        int n = (int)nt.size();
        std::vector<double> ldf(n);
        for (int i = 0; i < n; ++i) ldf[i] = std::log(nd[i]);

        if (t <= nt[0]) return std::exp(ldf[0] * t / nt[0]);
        if (t >= nt[n-1]) {
            double slope = (ldf[n-1] - ldf[n-2]) / (nt[n-1] - nt[n-2]);
            return std::exp(ldf[n-1] + slope * (t - nt[n-1]));
        }
        for (int i = 1; i < n; ++i) {
            if (t <= nt[i]) {
                if (i == 1) {
                    double alpha = (t - nt[0]) / (nt[1] - nt[0]);
                    return std::exp(ldf[0] + alpha * (ldf[1] - ldf[0]));
                }
                double w = (t - nt[i-1]) / (nt[i] - nt[i-1]);
                auto Ql = quadCoeffs(nt[i-2], ldf[i-2], nt[i-1], ldf[i-1], nt[i], ldf[i]);
                double val;
                if (i < n-1) {
                    auto Qr = quadCoeffs(nt[i-1], ldf[i-1], nt[i], ldf[i], nt[i+1], ldf[i+1]);
                    val = (1.0-w)*evalQuad(Ql,t) + w*evalQuad(Qr,t);
                } else {
                    val = evalQuad(Ql, t);
                }
                return std::exp(val);
            }
        }
        throw std::runtime_error("AvgQuadInterp: unreachable state");
    }

    double d_lnDF_dDFk(double t, int k,
                        const std::vector<double>& nt,
                        const std::vector<double>& nd) const override {
        int n = (int)nt.size();

        if (t <= nt[0]) return (k == 0) ? (t / nt[0]) / nd[0] : 0.0;
        if (t >= nt[n-1]) {
            if (k == n-1) return (1.0 + (t-nt[n-1])/(nt[n-1]-nt[n-2])) / nd[n-1];
            if (k == n-2) return (-(t-nt[n-1])/(nt[n-1]-nt[n-2])) / nd[n-2];
            return 0.0;
        }

        for (int i = 1; i < n; ++i) {
            if (t <= nt[i]) {
                if (i == 1) {
                    double alpha = (t - nt[0]) / (nt[1] - nt[0]);
                    if (k == 0) return (1.0 - alpha) / nd[0];
                    if (k == 1) return alpha / nd[1];
                    return 0.0;
                }

                double w = (t - nt[i-1]) / (nt[i] - nt[i-1]);

                auto dQuad_dLogDFj = [&](int qNode0, int qNode1, int qNode2, int j) -> double {
                    double ta = nt[qNode0], tb = nt[qNode1], tc = nt[qNode2];
                    double Amat[3][3] = {{ta*ta,ta,1},{tb*tb,tb,1},{tc*tc,tc,1}};
                    double rhs[3] = {0.0, 0.0, 0.0};
                    if      (j == qNode0) rhs[0] = 1.0;
                    else if (j == qNode1) rhs[1] = 1.0;
                    else if (j == qNode2) rhs[2] = 1.0;
                    else return 0.0;
                    auto dc = solve3x3(Amat, rhs);
                    return dc[0]*t*t + dc[1]*t + dc[2];
                };

                double d_lnDF_dlogDFk = 0.0;
                d_lnDF_dlogDFk += (1.0 - w) * dQuad_dLogDFj(i-2, i-1, i, k);
                if (i < n-1)
                    d_lnDF_dlogDFk += w * dQuad_dLogDFj(i-1, i, i+1, k);

                return d_lnDF_dlogDFk / nd[k];
            }
        }
        return 0.0;
    }
};

// ============================================================
// INSTRUMENT INTERFACE & IMPLEMENTATIONS
// ============================================================

class Instrument {
public:
    virtual ~Instrument() = default;
    virtual double parRateResidual(
        double df_terminal,
        double t_terminal,
        double market_rate,
        const std::vector<double>& nodes_t,
        const std::vector<double>& nodes_df,
        const InterpolationMethod& interp) const = 0;
};

class CashInstrument : public Instrument {
public:
    double parRateResidual(
        double df_terminal, double t_terminal, double market_rate,
        const std::vector<double>&, const std::vector<double>&,
        const InterpolationMethod&) const override {
        double implied_rate = (1.0/df_terminal - 1.0) / dcf(t_terminal);
        return implied_rate - market_rate;
    }
};

class SwapInstrument : public Instrument {
public:
    double parRateResidual(
        double df_terminal, double t_terminal, double market_rate,
        const std::vector<double>& nodes_t,
        const std::vector<double>& nodes_df,
        const InterpolationMethod& interp) const override {

        std::vector<double> trial_t  = nodes_t;
        std::vector<double> trial_df = nodes_df;
        trial_t.push_back(t_terminal);
        trial_df.push_back(df_terminal);

        auto getDF = [&](double t) {
            return interp.getDF(t, trial_t, trial_df);
        };

        std::vector<double> pay_dates;
        if (t_terminal <= 180.0) {
            pay_dates.push_back(t_terminal);
        } else {
            for (double t = 180.0; t < t_terminal - 1e-9; t += 180.0)
                pay_dates.push_back(t);
            pay_dates.push_back(t_terminal);
        }

        double den = 0.0;
        double prev = 0.0;
        for (double tj : pay_dates) {
            den += getDF(tj) * dcf(tj, prev);
            prev = tj;
        }
        return (1.0 - df_terminal) / den - market_rate;
    }

    std::vector<double> dParRate_dDFm(
        double df_terminal, double t_terminal,
        const std::vector<double>& nodes_t,
        const std::vector<double>& nodes_df,
        const InterpolationMethod& interp) const {

        std::vector<double> trial_t  = nodes_t;
        std::vector<double> trial_df = nodes_df;
        trial_t.push_back(t_terminal);
        trial_df.push_back(df_terminal);
        int n = (int)trial_t.size();

        std::vector<double> pay_dates;
        if (t_terminal <= 180.0) {
            pay_dates.push_back(t_terminal);
        } else {
            for (double t = 180.0; t < t_terminal - 1e-9; t += 180.0)
                pay_dates.push_back(t);
            pay_dates.push_back(t_terminal);
        }

        double den = 0.0;
        double prev = 0.0;
        for (double tj : pay_dates) {
            den += interp.getDF(tj, trial_t, trial_df) * dcf(tj, prev);
            prev = tj;
        }
        double num = 1.0 - df_terminal;

        std::vector<double> result(n, 0.0);
        for (int m = 0; m < n; ++m) {
            double d_num = -interp.dDF_dDFk(t_terminal, m, trial_t, trial_df);
            double d_den = 0.0;
            prev = 0.0;
            for (double tj : pay_dates) {
                d_den += interp.dDF_dDFk(tj, m, trial_t, trial_df) * dcf(tj, prev);
                prev = tj;
            }
            result[m] = (d_num * den - num * d_den) / (den * den);
        }
        return result;
    }
};

// ============================================================
// INTEREST RATE CURVE WRAPPER
// ============================================================

class IRCurve {
public:
    std::vector<double> nodes_t;
    std::vector<double> nodes_df;
    const InterpolationMethod* interp;

    IRCurve(const InterpolationMethod* interp_) : interp(interp_) {}

    double getDF(double t) const {
        return interp->getDF(t, nodes_t, nodes_df);
    }

    void addNode(double t, double df) {
        nodes_t.push_back(t);
        nodes_df.push_back(df);
    }

    std::vector<double> compute_dPV_dDFk(
        const std::vector<double>& fixed_dates,
        const std::vector<double>& float_dates,
        double notional, double r_fixed) const {

        int n = (int)nodes_t.size();
        std::vector<double> dPV(n, 0.0);

        // Fixed Leg Derivatives
        {
            double prev = 0.0;
            for (double tj : fixed_dates) {
                double d = dcf(tj, prev);
                for (int k = 0; k < n; ++k)
                    dPV[k] -= notional * r_fixed * d * interp->dDF_dDFk(tj, k, nodes_t, nodes_df);
                prev = tj;
            }
        }

        // Float Leg Derivatives
        {
            double prev = 0.0;
            for (double tj : float_dates) {
                for (int k = 0; k < n; ++k) {
                    double d_prev = (prev > 1e-9) ? interp->dDF_dDFk(prev, k, nodes_t, nodes_df) : 0.0;
                    double d_tj   = interp->dDF_dDFk(tj, k, nodes_t, nodes_df);
                    dPV[k] += notional * (d_prev - d_tj);
                }
                prev = tj;
            }
        }
        return dPV;
    }
};

class CurveBuilder {
public:
    static IRCurve build(
        const std::vector<CurveNode>& market_data,
        const Instrument& instrument,
        const InterpolationMethod& interp) {

        IRCurve curve(&interp);

        for (const auto& node : market_data) {
            double T    = node.t;
            double rate = node.rate;

            auto residual = [&](double df) {
                return instrument.parRateResidual(
                    df, T, rate, curve.nodes_t, curve.nodes_df, interp);
            };

            double df_sol;
            try {
                df_sol = brent(residual, 1e-6, 1.0 - 1e-9);
            } catch (...) {
                df_sol = 1.0 / (1.0 + rate * dcf(T));
            }
            curve.addNode(T, df_sol);
        }
        return curve;
    }
};

// ============================================================
// PRICING SYSTEM
// ============================================================

struct SwapSpec {
    double notional;
    double r_fixed;
    double maturity_days;
    int    fixed_freq_days;
    int    float_freq_days;
};

struct SwapResult {
    double pv;
    double par_rate;
};

static std::vector<double> makeSchedule(int freq_days, double maturity_days) {
    std::vector<double> dates;
    for (double t = freq_days; t <= maturity_days + 1e-9; t += freq_days)
        dates.push_back(t);
    return dates;
}

class SwapPricer {
public:
    static SwapResult price(const SwapSpec& s, const IRCurve& curve) {
        auto fd = makeSchedule(s.fixed_freq_days, s.maturity_days);
        auto fl = makeSchedule(s.float_freq_days, s.maturity_days);

        double pv_float = 0.0, pv_fixed = 0.0;
        {
            double prev = 0.0;
            for (double tj : fl) {
                double dfprev = (prev > 1e-9) ? curve.getDF(prev) : 1.0;
                pv_float += s.notional * (dfprev - curve.getDF(tj));
                prev = tj;
            }
        }
        {
            double prev = 0.0;
            for (double tj : fd) {
                pv_fixed += s.notional * s.r_fixed * dcf(tj, prev) * curve.getDF(tj);
                prev = tj;
            }
        }

        double num = 0.0, den = 0.0;
        {
            double prev = 0.0;
            for (double tj : fl) {
                double dfprev = (prev > 1e-9) ? curve.getDF(prev) : 1.0;
                double dftj   = curve.getDF(tj);
                double d = dcf(tj, prev);
                num += (dfprev/dftj - 1.0) / d * dftj * d;
                prev = tj;
            }
        }
        {
            double prev = 0.0;
            for (double tj : fd) {
                den += dcf(tj, prev) * curve.getDF(tj);
                prev = tj;
            }
        }
        return { pv_float - pv_fixed, num / den };
    }
};

// ============================================================
// RISK PROFILES (Implicit Function Theorem Base)
// ============================================================

static std::vector<double> cashCurveRisk(
    const IRCurve& curve,
    const std::vector<double>& cash_rates,
    const SwapSpec& s) {

    auto fd = makeSchedule(s.fixed_freq_days,  s.maturity_days);
    auto fl = makeSchedule(s.float_freq_days, s.maturity_days);
    auto dPV_dDF = curve.compute_dPV_dDFk(fd, fl, s.notional, s.r_fixed);

    int n = (int)curve.nodes_t.size();
    std::vector<double> risk(n);
    for (int i = 0; i < n; ++i) {
        double T = curve.nodes_t[i];
        double r = cash_rates[i];
        double d = dcf(T);
        double dDF_dc = -d / ((1.0 + r*d) * (1.0 + r*d));
        risk[i] = dPV_dDF[i] * dDF_dc;
    }
    return risk;
}

static std::vector<double> swapCurveRiskAnalytical(
    const std::vector<CurveNode>& market_data,
    const IRCurve& curve,
    const SwapInstrument& instr,
    const InterpolationMethod& interp,
    const SwapSpec& s) {

    int N = (int)market_data.size();
    auto fd = makeSchedule(s.fixed_freq_days,  s.maturity_days);
    auto fl = makeSchedule(s.float_freq_days, s.maturity_days);
    auto dPV_dDF = curve.compute_dPV_dDFk(fd, fl, s.notional, s.r_fixed);

    std::vector<std::vector<double>> J(N, std::vector<double>(N, 0.0));

    for (int k = 0; k < N; ++k) {
        double T = curve.nodes_t[k];
        double r = market_data[k].rate;

        if (T <= 180.0) {
            double d = T / 360.0;
            J[k][k] = -d / ((1.0 + r*d) * (1.0 + r*d));
        } else {
            auto dPSR_dDF = instr.dParRate_dDFm(
                curve.nodes_df[k], T,
                std::vector<double>(curve.nodes_t.begin(), curve.nodes_t.begin() + k),
                std::vector<double>(curve.nodes_df.begin(), curve.nodes_df.begin() + k),
                interp);

            double dF_dDFk = dPSR_dDF[k];

            for (int i = 0; i <= k; ++i) {
                double numerator = (i == k) ? 1.0 : 0.0;
                for (int j = 0; j < k; ++j)
                    numerator -= dPSR_dDF[j] * J[j][i];
                J[k][i] = numerator / dF_dDFk;
            }
        }
    }

    std::vector<double> risk(N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            risk[i] += dPV_dDF[k] * J[k][i];

    return risk;
}

// ============================================================
// DATA ETL / INPUT PARSING ENGINE
// ============================================================

struct InputData {
    int N;
    std::vector<std::string> maturity_strs;
    std::vector<double> cash_rates;
    std::vector<double> swap_rates;
    double query_t;
    SwapSpec swap;
};

static InputData parseInput(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open or find target file: " + filename);

    InputData inp;
    std::string line;

    std::getline(f, line);
    if (!line.empty() && (unsigned char)line[0] == 0xEF) line = line.substr(3); // UTF-8 BOM clear
    line = sanitizeString(line);
    inp.N = std::stoi(line);

    for (int i = 0; i < inp.N; ++i) {
        std::getline(f, line);
        std::istringstream ss(line);
        std::string mat, cr, sr;
        std::getline(ss, mat, ',');
        std::getline(ss, cr,  ',');
        std::getline(ss, sr,  ',');
        
        mat = sanitizeString(mat);
        cr  = sanitizeString(cr);
        sr  = sanitizeString(sr);

        inp.maturity_strs.push_back(mat);
        inp.cash_rates.push_back(cr.empty() ? 0.0 : std::stod(cr) / 100.0);
        inp.swap_rates.push_back(sr.empty() ? 0.0 : std::stod(sr) / 100.0);
    }

    std::getline(f, line);
    inp.query_t = std::stod(sanitizeString(line));

    std::getline(f, line);
    {
        std::istringstream ss(line);
        std::string rf, mat, ff, flo;
        std::getline(ss, rf,  ',');
        std::getline(ss, mat, ',');
        std::getline(ss, ff,  ',');
        std::getline(ss, flo, ',');
        
        inp.swap.notional        = 100.0;
        inp.swap.r_fixed         = std::stod(sanitizeString(rf)) / 100.0;
        inp.swap.maturity_days   = maturityToDays(sanitizeString(mat));
        inp.swap.fixed_freq_days = freqToDays(sanitizeString(ff));
        inp.swap.float_freq_days = freqToDays(sanitizeString(flo));
    }
    return inp;
}

// ============================================================
// MAIN EXECUTION CONTEXT
// ============================================================

int main(int argc, char* argv[]) {
    std::string input_file  = (argc > 1) ? argv[1] : "Input.csv";
    std::string output_file = (argc > 2) ? argv[2] : "Output.csv";

    InputData inp;
    try {
        inp = parseInput(input_file);
    } catch (const std::exception& e) {
        std::cerr << "Data Pipeline Error: " << e.what() << "\n";
        return 1;
    }

    std::vector<CurveNode> cash_nodes, swap_nodes;
    for (int i = 0; i < inp.N; ++i) {
        double t = maturityToDays(inp.maturity_strs[i]);
        cash_nodes.push_back({t, inp.cash_rates[i]});
        swap_nodes.push_back({t, inp.swap_rates[i]});
    }

    LinearInterp  linear;
    AvgQuadInterp aq;
    CashInstrument cash_instr;
    SwapInstrument swap_instr;

    IRCurve cash_lin = CurveBuilder::build(cash_nodes, cash_instr, linear);
    IRCurve cash_aq  = CurveBuilder::build(cash_nodes, cash_instr, aq);
    IRCurve swap_lin = CurveBuilder::build(swap_nodes, swap_instr, linear);
    IRCurve swap_aq  = CurveBuilder::build(swap_nodes, swap_instr, aq);

    // Q1 Metric Evaluations
    double q1a = cash_lin.getDF(inp.query_t);
    double q1b = cash_aq.getDF(inp.query_t);
    double q1c = swap_lin.getDF(inp.query_t);
    double q1d = swap_aq.getDF(inp.query_t);

    // Q2.1 Pricing Matrix Evaluations
    auto r21a = SwapPricer::price(inp.swap, cash_lin);
    auto r21b = SwapPricer::price(inp.swap, cash_aq);
    auto r21c = SwapPricer::price(inp.swap, swap_lin);
    auto r21d = SwapPricer::price(inp.swap, swap_aq);

    // Q2.2 Analytical Sensitivity Matrix Configurations
    auto q22a = cashCurveRisk(cash_lin, inp.cash_rates, inp.swap);
    auto q22b = cashCurveRisk(cash_aq,  inp.cash_rates, inp.swap);
    auto q22c = swapCurveRiskAnalytical(swap_nodes, swap_lin, swap_instr, linear, inp.swap);
    auto q22d = swapCurveRiskAnalytical(swap_nodes, swap_aq,  swap_instr, aq,     inp.swap);

    // Flush structured data clean to disk
    std::ofstream out(output_file, std::ios::trunc);
    if (!out) { std::cerr << "Cannot write output destination payload.\n"; return 1; }
    out << std::fixed << std::setprecision(8);

    out << q1a << "," << q1b << "," << q1c << "," << q1d << "\n";
    out << r21a.pv << "," << r21b.pv << "," << r21c.pv << "," << r21d.pv << "\n";
    out << r21a.par_rate << "," << r21b.par_rate << "," << r21c.par_rate << "," << r21d.par_rate << "\n";
    for (int i = 0; i < inp.N; ++i)
        out << q22a[i] << "," << q22b[i] << "," << q22c[i] << "," << q22d[i] << "\n";

    out.flush();
    out.close();

    std::cout << "Engine Run Success. Final payload safely dumped to " << output_file << "\n";
    return 0;
}