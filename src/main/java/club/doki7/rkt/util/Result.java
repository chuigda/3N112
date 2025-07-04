package club.doki7.rkt.util;

public sealed interface Result<T, E> {
    final class Ok<T, E> implements Result<T, E> {
        public final T value;

        public Ok(T value) {
            this.value = value;
        }

        @Override
        public String toString() {
            return "Ok(" + value + ")";
        }
    }

    final class Err<T, E> implements Result<T, E> {
        public final E error;

        public Err(E error) {
            this.error = error;
        }

        @Override
        public String toString() {
            return "Err(" + error + ")";
        }
    }

    static <T, E> Result<T, E> ok(T value) {
        return new Ok<>(value);
    }

    static <T, E> Result<T, E> err(E error) {
        return new Err<>(error);
    }
}
