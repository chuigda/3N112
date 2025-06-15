package club.doki7.rkt.vk;

@FunctionalInterface
public interface IDisposeOnContext {
    void disposeOnContext(RenderContext cx);

    IDisposeOnContext POISON = _ -> {};
}
