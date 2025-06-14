package club.doki7.cg112.util;

import org.jetbrains.annotations.NotNull;

public final class ReadonlyRef<T> {
    public final @NotNull T value;

    public ReadonlyRef(@NotNull T value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return value.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        ReadonlyRef<?> that = (ReadonlyRef<?>) obj;
        return value.equals(that.value);
    }

    @Override
    public int hashCode() {
        return value.hashCode();
    }
}
