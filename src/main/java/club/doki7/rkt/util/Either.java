package club.doki7.rkt.util;

import org.jetbrains.annotations.NotNull;

public sealed interface Either<L, R> {
    final class Left<L, R> implements Either<L, R> {
        public final L left;

        public Left(L left) {
            this.left = left;
        }

        @Override
        public @NotNull String toString() {
            return "Left(" + left + ")";
        }
    }

    final class Right<L, R> implements Either<L, R> {
        public final R right;

        public Right(R right) {
            this.right = right;
        }

        @Override
        public @NotNull String toString() {
            return "Right(" + right + ")";
        }
    }
}
