{ ... }:
{
  flyingcircus.roles.ai-api-gateway.enable = true;
  flyingcircus.roles.ai-api-gateway.settings = {
    backend = [
      {
        type = "skvaider";
        url = "{{component.inference_url}}";
      }
    ];
    auth.admin_tokens = [ "{{component.token}}" ];
    server.directory = "/var/lib/skvaider";
    logging.log_level = "DEBUG";
    models = [
      {
        id = "tiny-gpt2";
        instances = 1;
        memory.ram = 1;
        task = "chat";
      }
    ];
  };
}
