// Быстрый оценщик 7 карт без больших таблиц.
//
// Карта — int 0..51: rank = card >> 2 (0..12 = 2..A), suit = card & 3 (0..3).
// evaluate7 возвращает сравнимый ранг руки: больше = сильнее. Старшие биты —
// категория (стрит-флеш > каре > ... > старшая карта), младшие — тай-брейки.

#pragma once

namespace pe {

enum Category {
    HIGH_CARD = 0, PAIR = 1, TWO_PAIR = 2, TRIPS = 3, STRAIGHT = 4,
    FLUSH = 5, FULL_HOUSE = 6, QUADS = 7, STRAIGHT_FLUSH = 8
};

// Высший ранг стрейта в 13-битной маске рангов, или -1.
inline int best_straight(int mask) {
    for (int high = 12; high >= 4; --high) {
        int need = 0b11111 << (high - 4);
        if ((mask & need) == need) return high;
    }
    // Колесо A-2-3-4-5: ранги A(12),2(0),3(1),4(2),5(3); старшая — пятёрка (rank 3).
    const int wheel = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
    if ((mask & wheel) == wheel) return 3;
    return -1;
}

// Категория + до пяти тай-брейк-рангов (старший важнее).
inline int make_score(int cat, int n1 = 0, int n2 = 0, int n3 = 0, int n4 = 0, int n5 = 0) {
    return (cat << 20) | (n1 << 16) | (n2 << 12) | (n3 << 8) | (n4 << 4) | n5;
}

inline int evaluate7(const int* cards) {
    int rc[13] = {0};        // счётчики рангов
    int sc[4] = {0};         // счётчики мастей
    int suit_mask[4] = {0};  // маска рангов по каждой масти
    int rank_mask = 0;
    for (int i = 0; i < 7; ++i) {
        int r = cards[i] >> 2, s = cards[i] & 3;
        rc[r]++;
        sc[s]++;
        suit_mask[s] |= (1 << r);
        rank_mask |= (1 << r);
    }

    int flush_suit = -1;
    for (int s = 0; s < 4; ++s)
        if (sc[s] >= 5) flush_suit = s;

    // Стрит-флеш.
    if (flush_suit >= 0) {
        int h = best_straight(suit_mask[flush_suit]);
        if (h >= 0) return make_score(STRAIGHT_FLUSH, h);
    }

    // Каре.
    for (int r = 12; r >= 0; --r)
        if (rc[r] == 4) {
            int k = -1;
            for (int x = 12; x >= 0; --x)
                if (x != r && rc[x] > 0) { k = x; break; }
            return make_score(QUADS, r, k);
        }

    // Тройки и пары по убыванию (каре уже отсеяно).
    int trips[13], nt = 0, pairs[13], np = 0;
    for (int r = 12; r >= 0; --r) {
        if (rc[r] == 3) trips[nt++] = r;
        else if (rc[r] == 2) pairs[np++] = r;
    }

    // Фулл-хаус: старшая тройка + (вторая тройка или старшая пара).
    if (nt >= 1) {
        int t = trips[0], p = -1;
        if (nt >= 2) p = trips[1];
        if (np >= 1 && pairs[0] > p) p = pairs[0];
        if (p >= 0) return make_score(FULL_HOUSE, t, p);
    }

    // Флеш (топ-5 рангов масти).
    if (flush_suit >= 0) {
        int top[5], n = 0;
        for (int r = 12; r >= 0 && n < 5; --r)
            if (suit_mask[flush_suit] & (1 << r)) top[n++] = r;
        return make_score(FLUSH, top[0], top[1], top[2], top[3], top[4]);
    }

    // Стрейт.
    int sh = best_straight(rank_mask);
    if (sh >= 0) return make_score(STRAIGHT, sh);

    // Сет.
    if (nt >= 1) {
        int t = trips[0], k[2], n = 0;
        for (int r = 12; r >= 0 && n < 2; --r)
            if (r != t && rc[r] > 0) k[n++] = r;
        return make_score(TRIPS, t, k[0], k[1]);
    }

    // Две пары.
    if (np >= 2) {
        int p1 = pairs[0], p2 = pairs[1], k = -1;
        for (int r = 12; r >= 0; --r)
            if (r != p1 && r != p2 && rc[r] > 0) { k = r; break; }
        return make_score(TWO_PAIR, p1, p2, k);
    }

    // Пара.
    if (np == 1) {
        int p = pairs[0], k[3], n = 0;
        for (int r = 12; r >= 0 && n < 3; --r)
            if (r != p && rc[r] > 0) k[n++] = r;
        return make_score(PAIR, p, k[0], k[1], k[2]);
    }

    // Старшая карта (топ-5).
    int top[5], n = 0;
    for (int r = 12; r >= 0 && n < 5; --r)
        if (rc[r] > 0) top[n++] = r;
    return make_score(HIGH_CARD, top[0], top[1], top[2], top[3], top[4]);
}

}  // namespace pe
