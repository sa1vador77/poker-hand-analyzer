// Покерная математика (C++/nanobind): эквити героя.
//
// equity_vs   — против ИЗВЕСТНЫХ рук оппонентов: неизвестен только борд, поэтому
//               перебираем все его завершения точно (или Монте-Карло, если их много).
// equity_random — против N оппонентов со СЛУЧАЙНЫМИ руками: Монте-Карло по неразданной
//               колоде. Питоний фасад — в `poker_analyzer.engine.equity`.

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

#include "evaluator.hpp"

namespace nb = nanobind;

namespace {

constexpr long long EXACT_CAP = 2'000'000;  // выше — переходим на Монте-Карло

// Быстрый PRNG (xorshift64*) — для Монте-Карло этого достаточно.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    inline uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    }
    inline int below(int n) { return static_cast<int>(next() % static_cast<uint64_t>(n)); }
};

struct Accum {
    double win = 0, tie = 0, eq = 0;
    long long total = 0;
};

inline long long n_choose_r(int n, int r) {
    if (r < 0 || r > n) return 0;
    r = std::min(r, n - r);
    long long res = 1;
    for (int i = 0; i < r; ++i) res = res * (n - i) / (i + 1);
    return res;
}

// Доля банка героя на одном раскладе (борд из 5 карт уже собран).
inline void showdown(const int hero[2], const std::vector<std::array<int, 2>>& opps,
                     const int board5[5], Accum& a) {
    int h7[7] = {hero[0], hero[1], board5[0], board5[1], board5[2], board5[3], board5[4]};
    int hs = pe::evaluate7(h7);

    int os[24];
    int no = static_cast<int>(opps.size());
    int best = -1;
    for (int o = 0; o < no; ++o) {
        int e7[7] = {opps[o][0], opps[o][1], board5[0], board5[1], board5[2], board5[3], board5[4]};
        os[o] = pe::evaluate7(e7);
        if (os[o] > best) best = os[o];
    }

    a.total++;
    if (hs > best) {
        a.win += 1.0;
        a.eq += 1.0;
    } else if (hs == best) {
        int k = 1;  // герой + столько же оппонентов с тем же рангом
        for (int o = 0; o < no; ++o)
            if (os[o] == hs) ++k;
        a.tie += 1.0;
        a.eq += 1.0 / k;
    }
}

// Точный перебор завершений борда.
void enum_board(const std::vector<int>& avail, int pos, int start, int board5[5],
                const int hero[2], const std::vector<std::array<int, 2>>& opps, Accum& a) {
    if (pos == 5) {
        showdown(hero, opps, board5, a);
        return;
    }
    for (int i = start; i < static_cast<int>(avail.size()); ++i) {
        board5[pos] = avail[i];
        enum_board(avail, pos + 1, i + 1, board5, hero, opps, a);
    }
}

std::tuple<double, double, double> finalize(const Accum& a) {
    if (a.total == 0) return {0.0, 0.0, 0.0};
    return {a.win / a.total, a.tie / a.total, a.eq / a.total};
}

// Карты, занятые героем/бордом/известными оппонентами -> оставшаяся колода.
std::vector<int> remaining_deck(const bool used[52]) {
    std::vector<int> avail;
    for (int c = 0; c < 52; ++c)
        if (!used[c]) avail.push_back(c);
    return avail;
}

// Точный перебор назначений: одна непротиворечивая рука из диапазона каждого оппонента,
// затем завершения борда. used[] держит карты героя/борда/уже выбранных оппонентов.
void enum_assign(const std::vector<std::vector<std::array<int, 2>>>& ranges, int o, bool used[52],
                 std::vector<std::array<int, 2>>& chosen, const int hero[2],
                 const std::vector<int>& board, int nfixed, Accum& a) {
    if (o == static_cast<int>(ranges.size())) {
        std::vector<int> avail = remaining_deck(used);
        int board5[5];
        for (int i = 0; i < nfixed; ++i) board5[i] = board[i];
        enum_board(avail, nfixed, 0, board5, hero, chosen, a);
        return;
    }
    for (const auto& h : ranges[o]) {
        if (used[h[0]] || used[h[1]]) continue;  // карта занята героем/бордом/другим оппонентом
        used[h[0]] = used[h[1]] = true;
        chosen.push_back(h);
        enum_assign(ranges, o + 1, used, chosen, hero, board, nfixed, a);
        chosen.pop_back();
        used[h[0]] = used[h[1]] = false;
    }
}

}  // namespace

std::tuple<double, double, double> equity_vs(std::vector<int> hero, std::vector<int> board,
                                             std::vector<std::vector<int>> opp_in,
                                             long long iterations, uint64_t seed) {
    if (opp_in.empty()) return {1.0, 0.0, 1.0};  // без оппонентов герой забирает банк

    bool used[52] = {false};
    used[hero[0]] = used[hero[1]] = true;
    for (int c : board) used[c] = true;
    std::vector<std::array<int, 2>> opps;
    for (auto& h : opp_in) {
        opps.push_back({h[0], h[1]});
        used[h[0]] = used[h[1]] = true;
    }
    std::vector<int> avail = remaining_deck(used);

    int nfixed = static_cast<int>(board.size());
    int need = 5 - nfixed;
    int board5[5];
    for (int i = 0; i < nfixed; ++i) board5[i] = board[i];
    int hero2[2] = {hero[0], hero[1]};

    Accum a;
    if (n_choose_r(static_cast<int>(avail.size()), need) <= EXACT_CAP) {
        enum_board(avail, nfixed, 0, board5, hero2, opps, a);
    } else {
        Rng rng(seed);
        int m = static_cast<int>(avail.size());
        for (long long it = 0; it < iterations; ++it) {
            for (int i = 0; i < need; ++i) std::swap(avail[i], avail[i + rng.below(m - i)]);
            for (int i = 0; i < need; ++i) board5[nfixed + i] = avail[i];
            showdown(hero2, opps, board5, a);
        }
    }
    return finalize(a);
}

std::tuple<double, double, double> equity_random(std::vector<int> hero, std::vector<int> board,
                                                 int num_opp, long long iterations,
                                                 uint64_t seed) {
    bool used[52] = {false};
    used[hero[0]] = used[hero[1]] = true;
    for (int c : board) used[c] = true;
    std::vector<int> avail = remaining_deck(used);

    int nfixed = static_cast<int>(board.size());
    int need_board = 5 - nfixed;
    int need = num_opp * 2 + need_board;
    int board5[5];
    for (int i = 0; i < nfixed; ++i) board5[i] = board[i];
    int hero2[2] = {hero[0], hero[1]};

    Accum a;
    Rng rng(seed);
    int m = static_cast<int>(avail.size());
    std::vector<std::array<int, 2>> opps(num_opp);
    for (long long it = 0; it < iterations; ++it) {
        for (int i = 0; i < need; ++i) std::swap(avail[i], avail[i + rng.below(m - i)]);
        for (int i = 0; i < need_board; ++i) board5[nfixed + i] = avail[i];
        for (int o = 0; o < num_opp; ++o) {
            opps[o][0] = avail[need_board + 2 * o];
            opps[o][1] = avail[need_board + 2 * o + 1];
        }
        showdown(hero2, opps, board5, a);
    }
    return finalize(a);
}

std::tuple<double, double, double, long long, bool> equity_vs_ranges(
    std::vector<int> hero, std::vector<int> board,
    std::vector<std::vector<std::vector<int>>> ranges_in, long long iterations,
    long long exact_cap, uint64_t seed) {
    int nopp = static_cast<int>(ranges_in.size());
    if (nopp == 0) return {1.0, 0.0, 1.0, 0, false};  // без оппонентов герой забирает банк

    bool base_used[52] = {false};
    base_used[hero[0]] = base_used[hero[1]] = true;
    for (int c : board) base_used[c] = true;

    // Выкидываем из диапазонов руки, заблокированные картами героя/борда.
    std::vector<std::vector<std::array<int, 2>>> ranges(nopp);
    for (int o = 0; o < nopp; ++o)
        for (auto& h : ranges_in[o])
            if (!base_used[h[0]] && !base_used[h[1]]) ranges[o].push_back({h[0], h[1]});
    for (auto& r : ranges)
        if (r.empty()) return {0.0, 0.0, 0.0, 0, false};  // диапазон целиком заблокирован — раскладов нет

    int nfixed = static_cast<int>(board.size());
    int need = 5 - nfixed;
    int hero2[2] = {hero[0], hero[1]};

    // Оценка работы точного перебора: (произв. размеров диапазонов) × (завершения борда).
    // Конфликты игнорируются — это оценка сверху, поэтому переключение консервативно.
    double work = 1.0;
    for (auto& r : ranges) work *= static_cast<double>(r.size());
    int avail_after = 52 - 2 - nfixed - 2 * nopp;
    work *= static_cast<double>(n_choose_r(std::max(avail_after, 0), need));

    Accum a;
    bool was_mc = work > static_cast<double>(exact_cap);
    if (!was_mc) {  // мало — считаем точно
        bool used[52];
        std::copy(base_used, base_used + 52, used);
        std::vector<std::array<int, 2>> chosen;
        chosen.reserve(nopp);
        enum_assign(ranges, 0, used, chosen, hero2, board, nfixed, a);
    } else {  // взрывается — Монте-Карло
        Rng rng(seed);
        std::vector<std::array<int, 2>> opps(nopp);
        int board5[5];
        for (int i = 0; i < nfixed; ++i) board5[i] = board[i];
        for (long long it = 0; it < iterations; ++it) {
            bool used[52];
            std::copy(base_used, base_used + 52, used);
            bool ok = true;
            for (int o = 0; o < nopp; ++o) {  // сдаём по руке из каждого диапазона
                const auto& r = ranges[o];
                int rs = static_cast<int>(r.size());
                bool placed = false;
                for (int t = 0; t < 256; ++t) {  // rejection: ищем непротиворечивую руку
                    const auto& h = r[rng.below(rs)];
                    if (!used[h[0]] && !used[h[1]]) {
                        opps[o] = h;
                        used[h[0]] = used[h[1]] = true;
                        placed = true;
                        break;
                    }
                }
                if (!placed) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;  // не удалось расставить руки без конфликта — пропускаем сэмпл
            std::vector<int> avail = remaining_deck(used);
            int m = static_cast<int>(avail.size());
            for (int i = 0; i < need; ++i) std::swap(avail[i], avail[i + rng.below(m - i)]);
            for (int i = 0; i < need; ++i) board5[nfixed + i] = avail[i];
            showdown(hero2, opps, board5, a);
        }
    }
    auto [w, t, e] = finalize(a);
    return {w, t, e, a.total, was_mc};  // n_eff = a.total (при MC — число удавшихся сэмплов)
}

// Эквити героя против КАЖДОГО комбо по отдельности (на текущем борде). Для слоя советов:
// по этим числам строится fold equity и эквити-при-колле при выборе размера ставки.
// Постфлоп доезд короткий (≤ C(45,2)) → точный перебор; иначе Монте-Карло. Заблокированное
// картами героя/борда комбо → -1.0 (вызывающая сторона его пропускает).
std::vector<double> hero_equity_vs_each(std::vector<int> hero, std::vector<int> board,
                                        std::vector<std::vector<int>> combos, long long iterations,
                                        long long exact_cap, uint64_t seed) {
    bool base_used[52] = {false};
    base_used[hero[0]] = base_used[hero[1]] = true;
    for (int c : board) base_used[c] = true;
    int nfixed = static_cast<int>(board.size());
    int need = 5 - nfixed;
    int hero2[2] = {hero[0], hero[1]};
    bool do_exact = n_choose_r(52 - 2 - nfixed - 2, need) <= exact_cap;

    std::vector<double> out;
    out.reserve(combos.size());
    Rng rng(seed);
    for (auto& cb : combos) {
        if (base_used[cb[0]] || base_used[cb[1]]) {
            out.push_back(-1.0);  // комбо заблокировано картами героя/борда
            continue;
        }
        bool used[52];
        std::copy(base_used, base_used + 52, used);
        used[cb[0]] = used[cb[1]] = true;
        std::vector<int> avail = remaining_deck(used);
        std::vector<std::array<int, 2>> opps = {{cb[0], cb[1]}};
        int board5[5];
        for (int i = 0; i < nfixed; ++i) board5[i] = board[i];
        Accum a;
        if (do_exact) {
            enum_board(avail, nfixed, 0, board5, hero2, opps, a);
        } else {
            int m = static_cast<int>(avail.size());
            for (long long it = 0; it < iterations; ++it) {
                for (int i = 0; i < need; ++i) std::swap(avail[i], avail[i + rng.below(m - i)]);
                for (int i = 0; i < need; ++i) board5[nfixed + i] = avail[i];
                showdown(hero2, opps, board5, a);
            }
        }
        out.push_back(a.total ? a.eq / a.total : 0.0);
    }
    return out;
}

// Эквити КАЖДОГО комбо против ОДНОЙ случайной руки на борде (Монте-Карло). Мерило силы
// руки на доске для сужения диапазона (мейд-руки и сильные дро → высокое, воздух → низкое).
std::vector<double> equity_each_vs_random(std::vector<int> board,
                                          std::vector<std::vector<int>> combos,
                                          long long iterations, uint64_t seed) {
    bool base_used[52] = {false};
    for (int c : board) base_used[c] = true;
    int nfixed = static_cast<int>(board.size());
    int need_board = 5 - nfixed;
    int need = 2 + need_board;  // случайная рука оппонента (2) + завершение борда

    std::vector<double> out;
    out.reserve(combos.size());
    Rng rng(seed);
    for (auto& cb : combos) {
        if (base_used[cb[0]] || base_used[cb[1]]) {
            out.push_back(-1.0);
            continue;
        }
        bool used[52];
        std::copy(base_used, base_used + 52, used);
        used[cb[0]] = used[cb[1]] = true;
        std::vector<int> avail = remaining_deck(used);
        int m = static_cast<int>(avail.size());
        int hero2[2] = {cb[0], cb[1]};
        int board5[5];
        for (int i = 0; i < nfixed; ++i) board5[i] = board[i];
        std::vector<std::array<int, 2>> opps(1);
        Accum a;
        for (long long it = 0; it < iterations; ++it) {
            for (int i = 0; i < need; ++i) std::swap(avail[i], avail[i + rng.below(m - i)]);
            for (int i = 0; i < need_board; ++i) board5[nfixed + i] = avail[i];
            opps[0][0] = avail[need_board];
            opps[0][1] = avail[need_board + 1];
            showdown(hero2, opps, board5, a);
        }
        out.push_back(a.total ? a.eq / a.total : 0.0);
    }
    return out;
}

// --- Классификация комбо на борде (для слоя советов: made/draw/air, ауты) -----
namespace {

// Категория готовой руки (pe::Category) по счётчикам рангов/мастей — для 5..7 карт.
int made_category_counts(const int rc[13], const int sc[4], const int suit_mask[4],
                         int rank_mask) {
    int flush_suit = -1;
    for (int s = 0; s < 4; ++s)
        if (sc[s] >= 5) flush_suit = s;
    if (flush_suit >= 0 && pe::best_straight(suit_mask[flush_suit]) >= 0) return pe::STRAIGHT_FLUSH;
    int quad = -1, trips = 0, pairs = 0;
    for (int r = 0; r < 13; ++r) {
        if (rc[r] == 4) quad = r;
        else if (rc[r] == 3) ++trips;
        else if (rc[r] == 2) ++pairs;
    }
    if (quad >= 0) return pe::QUADS;
    if (trips >= 1 && (trips >= 2 || pairs >= 1)) return pe::FULL_HOUSE;
    if (flush_suit >= 0) return pe::FLUSH;
    if (pe::best_straight(rank_mask) >= 0) return pe::STRAIGHT;
    if (trips >= 1) return pe::TRIPS;
    if (pairs >= 2) return pe::TWO_PAIR;
    if (pairs == 1) return pe::PAIR;
    return pe::HIGH_CARD;
}

// (made_category, flush_draw, oesd, gutshot, outs) для одного комбо на борде.
std::tuple<int, int, int, int, int> classify_one(const std::vector<int>& board, int c0, int c1) {
    int rc[13] = {0}, sc[4] = {0}, suit_mask[4] = {0}, rank_mask = 0;
    auto add = [&](int cm) {
        int r = cm >> 2, s = cm & 3;
        rc[r]++; sc[s]++; suit_mask[s] |= (1 << r); rank_mask |= (1 << r);
    };
    add(c0);
    add(c1);
    for (int b : board) add(b);

    int made = made_category_counts(rc, sc, suit_mask, rank_mask);

    bool has_flush = false;
    for (int s = 0; s < 4; ++s)
        if (sc[s] >= 5) has_flush = true;
    int flush_draw = 0;
    if (!has_flush)
        for (int s = 0; s < 4; ++s)
            if (sc[s] == 4) flush_draw = 1;  // ровно 4 одной масти — флеш-дро

    int completing = 0;  // сколько рангов достраивают стрейт (если его ещё нет)
    if (pe::best_straight(rank_mask) < 0) {
        for (int r = 0; r < 13; ++r) {
            if (rank_mask & (1 << r)) continue;
            if (pe::best_straight(rank_mask | (1 << r)) >= 0) ++completing;
        }
    }
    int oesd = completing >= 2 ? 1 : 0;      // двусторонний (или дабл-гатшот): ≥2 ранга
    int gutshot = completing == 1 ? 1 : 0;   // гатшот: ровно 1 ранг

    int outs = (flush_draw ? 9 : 0) + (oesd ? 8 : (gutshot ? 4 : 0));
    if (outs > 15) outs = 15;  // грубый потолок на пересечение флеш/стрейт аутов
    return {made, flush_draw, oesd, gutshot, outs};
}

}  // namespace

// По каждому комбо — (made_category, flush_draw, oesd, gutshot, outs); заблокированное
// бордом (или вырожденное) комбо → (-1, 0, 0, 0, 0). Вызывающая сторона строит value/bluff
// сплит и класс руки героя (made/draw/air) для реализации эквити.
std::vector<std::tuple<int, int, int, int, int>> classify_combos(
    std::vector<int> board, std::vector<std::vector<int>> combos) {
    bool on_board[52] = {false};
    for (int b : board) on_board[b] = true;
    std::vector<std::tuple<int, int, int, int, int>> out;
    out.reserve(combos.size());
    for (auto& c : combos) {
        if (c[0] == c[1] || on_board[c[0]] || on_board[c[1]])
            out.push_back({-1, 0, 0, 0, 0});
        else
            out.push_back(classify_one(board, c[0], c[1]));
    }
    return out;
}

NB_MODULE(_equity, m) {
    m.doc() = "Покерная математика (C++/nanobind): оценщик 7 карт и эквити героя.";
    m.def(
        "evaluate7", [](const std::vector<int>& c) { return pe::evaluate7(c.data()); },
        "Оценка 7 карт (int 0..51) -> сравнимый ранг (больше = сильнее).");
    // Тяжёлые функции ОТПУСКАЮТ GIL (аргументы уже сконвертированы в std::*, тел Python
    // не трогают): иначе многосекундный Монте-Карло в воркере замораживал главный поток
    // Tk — оверлей не отрисовывал даже «думаю…» и выглядел зависшим.
    m.def("equity_vs", &equity_vs, nb::call_guard<nb::gil_scoped_release>(),
          "Эквити героя против известных рук оппонентов (точный перебор борда).");
    m.def("equity_random", &equity_random, nb::call_guard<nb::gil_scoped_release>(),
          "Эквити героя против N оппонентов со случайными руками (Монте-Карло).");
    m.def("equity_vs_ranges", &equity_vs_ranges, nb::call_guard<nb::gil_scoped_release>(),
          "Эквити героя против диапазонов оппонентов (адаптивно: перебор / Монте-Карло).");
    m.def("hero_equity_vs_each", &hero_equity_vs_each, nb::call_guard<nb::gil_scoped_release>(),
          "Эквити героя против каждого комбо по отдельности на борде (-1 для заблокированных).");
    m.def("equity_each_vs_random", &equity_each_vs_random,
          nb::call_guard<nb::gil_scoped_release>(),
          "Эквити каждого комбо против случайной руки на борде (мерило силы для сужения).");
    m.def("classify_combos", &classify_combos, nb::call_guard<nb::gil_scoped_release>(),
          "Классификация комбо на борде: (made_category, flush_draw, oesd, gutshot, outs).");
}
