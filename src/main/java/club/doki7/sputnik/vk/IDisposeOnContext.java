package club.doki7.sputnik.vk;

@FunctionalInterface
public interface IDisposeOnContext {
    void disposeOnContext(RenderContext cx);

    IDisposeOnContext POISON = _ -> {};
}
