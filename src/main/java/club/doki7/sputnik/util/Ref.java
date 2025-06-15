package club.doki7.sputnik.util;

import java.util.Objects;

public final class Ref<T> {
    public T value;

    public Ref(T value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return value.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Ref<?> ref)) return false;
        return Objects.equals(value, ref.value);
    }

    @Override
    public int hashCode() {
        return Objects.hash(value);
    }
}
