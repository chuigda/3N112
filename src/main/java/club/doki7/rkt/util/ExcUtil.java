package club.doki7.rkt.util;

import org.jetbrains.annotations.NotNull;

import java.util.function.Consumer;
import java.util.function.Function;

public interface ExcUtil {
    @FunctionalInterface
    interface CheckedFunction<A, B, E extends Throwable> {
        B apply(A a) throws E;
    }

    @FunctionalInterface
    interface CheckedConsumer<A, E extends Throwable> {
        void consume(A a) throws E;
    }

    @SuppressWarnings("unchecked")
    static <A, B, E extends Throwable> Function<A, B>
    sneakyFunc(@NotNull CheckedFunction<A, B, ?> function) {
        var f = (CheckedFunction<A, B, RuntimeException>) function;
        return f::apply;
    }

    @SuppressWarnings("unchecked")
    static <A, E extends Throwable> Consumer<A>
    sneakyConsumer(@NotNull CheckedConsumer<A, ?> consumer) {
        var c = (CheckedConsumer<A, RuntimeException>) consumer;
        return c::consume;
    }
}
