//! Exact GPU predicates use modular determinants and mixed-radix reconstruction.
//! Ninety-six distinct primes >2^15 give modulus >2^1440, exceeding twice the
//! <2^1397 insphere bound for finite binary32 coordinates scaled by 2^149.
const count = 96;
fn isPrime(n: u32) bool {
    var d: u32 = 2;
    while (d * d <= n) : (d += 1) if (n % d == 0) return false;
    return true;
}
fn power(a: u32, exponent: u32, p: u32) u32 {
    var b: u64 = a % p;
    var e = exponent;
    var r: u64 = 1;
    while (e != 0) : (e >>= 1) {
        if (e & 1 != 0) r = r * b % p;
        b = b * b % p;
    }
    return @intCast(r);
}
pub const table = blk: {
    @setEvalBranchQuota(3000000);
    var result: [count + count * count + count * 256]u32 = @splat(0);
    var candidate: u32 = 65521;
    for (0..count) |i| {
        while (!isPrime(candidate)) : (candidate -= 2) {}
        result[i] = candidate;
        candidate -= 2;
    }
    for (0..count) |j| for (j + 1..count) |i| {
        result[count + j * count + i] = power(result[j], result[i] - 2, result[i]);
    };
    for (0..count) |i| {
        var value: u32 = 1;
        for (0..256) |j| {
            result[count + count * count + i * 256 + j] = value;
            value = value * 2 % result[i];
        }
    }
    break :blk result;
};
