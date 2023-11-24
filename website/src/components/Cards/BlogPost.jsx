const CARD_WIDTH = '300px'; // Example width, adjust as needed
const MARGIN = '10px'; // Example margin, adjust as needed

import arrow_ from "../../../static/img/arrow-narrow-up-right.svg";

const BlogPost = ({ imgUrl, tag, title, description }) => {
  return (
    <div
      className="relative shrink-0 cursor-pointer transition-transform hover:-translate-y-1 group"
      style={{
        width: CARD_WIDTH,
        marginRight: MARGIN,
      }}
    >
      <div className="relative">
        <img
          src={imgUrl}
          style={{
            marginBottom: '12px',
            height: '200px',
            width: '100%',
            borderRadius: '8px',
            objectFit: 'cover',
          }}
          alt={`An image for a blog post titled ${title}`}
        />
        <div
          style={{
            position: 'absolute',
            bottom: '8px',
            left: '8px',
            height: '24px',
            padding: '12px',
            backgroundColor: 'rgba(0, 0, 0, 0.6)',
            borderRadius: 'full',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <div
            style={{
              color: 'white',
              fontSize: '12px',
              fontWeight: '600',
              textTransform: 'uppercase',
            }}
          >
            {tag}
          </div>
        </div>
      </div>
      <div className="flex flex-row items-start">
        <div className="flex flex-col justify-start text-left">
          <p style={{ fontSize: '18px', fontWeight: '500' }}>{title}</p>
          <p style={{ fontSize: '14px', color: '#9CA3AF' }}>{description}</p>
        </div>
        <div
          className="group-hover:translate-x-1 group-hover:-translate-y-1 duration-500 ease-in-out transform"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flex: '1',
          }}
        >
          <img src={arrow_} alt="up-arrow" style={{ height: '100%', width: '100%' }} />
        </div>
      </div>
    </div>
  );
};

export default BlogPost;