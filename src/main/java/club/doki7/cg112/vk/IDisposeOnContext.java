package club.doki7.cg112.vk;

@FunctionalInterface
public interface IDisposeOnContext {
    void disposeOnContext(RenderContext cx);

    IDisposeOnContext POISON = _ -> {};
}
