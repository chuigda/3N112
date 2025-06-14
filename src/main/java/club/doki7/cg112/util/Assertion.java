package club.doki7.cg112.util;

public final class Assertion {
    public static final boolean assertionEnabled = Impl.assertionEnabled;

    @SuppressWarnings("AssertWithSideEffects")
    private static class Impl {
        static boolean assertionEnabled = false;
        static {
            assert assertionEnabled = true;
        }
    }
}
